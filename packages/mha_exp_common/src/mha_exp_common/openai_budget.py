"""Budget-aware OpenAI transport and per-execution credential lifecycle."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Literal

from mha_exp_common.batch import NonRetryableBatchError

PRICING_POLICY_VERSION = "openai-standard-2026-08-31-v2"
CONSERVATIVE_MAX_OUTPUT_TOKENS = 128_000


@dataclass(frozen=True)
class ModelPricing:
    """USD prices per million tokens for one OpenAI model."""

    input: Decimal
    cached_input: Decimal
    cache_write: Decimal
    output: Decimal
    long_context_threshold: int | None = None
    long_input_multiplier: Decimal = Decimal(1)
    long_output_multiplier: Decimal = Decimal(1)


_NANO = ModelPricing(
    input=Decimal("0.20"),
    cached_input=Decimal("0.02"),
    cache_write=Decimal("0.20"),
    output=Decimal("1.25"),
)
_MINI = ModelPricing(
    input=Decimal("0.75"),
    cached_input=Decimal("0.075"),
    cache_write=Decimal("0.75"),
    output=Decimal("4.50"),
)
_LUNA = ModelPricing(
    input=Decimal("0.20"),
    cached_input=Decimal("0.02"),
    cache_write=Decimal("0.25"),
    output=Decimal("1.20"),
    long_context_threshold=272_000,
    long_input_multiplier=Decimal(2),
    long_output_multiplier=Decimal("1.5"),
)
_GPT_5_3_CHAT = ModelPricing(
    input=Decimal("1.75"),
    cached_input=Decimal("0.175"),
    cache_write=Decimal("1.75"),
    output=Decimal("14.00"),
)
_GPT_5_4 = ModelPricing(
    input=Decimal("2.50"),
    cached_input=Decimal("0.25"),
    cache_write=Decimal("2.50"),
    output=Decimal("15.00"),
    long_context_threshold=272_000,
    long_input_multiplier=Decimal(2),
    long_output_multiplier=Decimal("1.5"),
)

MODEL_PRICING: Mapping[str, ModelPricing] = MappingProxyType(
    {
        "gpt-5.4-nano": _NANO,
        "gpt-5.4-nano-2026-03-17": _NANO,
        "gpt-5.4-mini": _MINI,
        "gpt-5.4-mini-2026-03-17": _MINI,
        "gpt-5.6-luna": _LUNA,
        "gpt-5.3-chat-latest": _GPT_5_3_CHAT,
        "gpt-5.4": _GPT_5_4,
        "gpt-5.4-2026-03-05": _GPT_5_4,
    }
)


class ProxyRequestError(RuntimeError):
    """Sanitized proxy failure annotated with Responses forwarding status."""

    def __init__(self, message: str, *, request_forwarded: bool) -> None:
        super().__init__(message)
        self.request_forwarded = request_forwarded


class BudgetExceededError(ProxyRequestError):
    """Raised when an instance has observed or retained an exhausted budget."""


class BudgetCheckError(ProxyRequestError):
    """Raised when filtered organization cost cannot be read safely."""


class UnknownModelPricingError(ProxyRequestError):
    """Raised before forwarding when a request model has no pricing policy."""


class UsageAccountingError(ProxyRequestError):
    """Raised after a response when its token usage cannot be accounted for."""

    def __init__(self, message: str, *, response: Any) -> None:
        super().__init__(message, request_forwarded=True)
        self.response = response

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"


class ClientInitializationError(RuntimeError):
    """Raised when either underlying OpenAI client cannot be constructed."""


class ClientCleanupError(RuntimeError):
    """Raised after all available OpenAI clients have been asked to close."""


class TemporaryCredentialError(RuntimeError):
    """Sanitized project service-account lifecycle failure."""

    def __init__(
        self,
        message: str,
        *,
        service_account_name: str,
        possible_leak: bool = False,
        cleanup_status: str = "not_created",
    ) -> None:
        super().__init__(message)
        self.service_account_name = service_account_name
        self.possible_leak = possible_leak
        self.cleanup_status = cleanup_status
        self.cleanup_failures: dict[str, dict[str, Any]] = {}


class TemporaryCredentialSecurityError(
    TemporaryCredentialError,
    NonRetryableBatchError,
):
    """Report a temporary credential that may remain accessible."""


@dataclass
class TemporaryExecutionCredential:
    """Host-only identifiers and clearable encoded key for one execution."""

    project_id: str
    service_account_id: str
    api_key_id: str
    service_account_name: str
    encoded_key: str = field(repr=False)
    cleanup_status: str = "pending"
    possible_leak: bool = False

    def clear_secret(self) -> None:
        """Discard the host holder's encoded API-key copy."""

        self.encoded_key = ""

    def metadata(self) -> dict[str, str | bool]:
        """Return secret-free lifecycle metadata."""

        return {
            "project_id": self.project_id,
            "service_account_id": self.service_account_id,
            "execution_api_key_id": self.api_key_id,
            "service_account_name": self.service_account_name,
            "cleanup_status": self.cleanup_status,
            "possible_leak": self.possible_leak,
        }


def _member(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    if hasattr(value, name):
        return getattr(value, name)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return dumped.get(name, default)
    return default


def _safe_error(error: BaseException) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(error).__name__}
    for name in ("status_code", "code"):
        value = getattr(error, name, None)
        if isinstance(value, (str, int)):
            result[name] = value
    return result


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _as_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(field_name)
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(field_name) from error
    if not numeric.is_finite() or numeric != numeric.to_integral_value():
        raise ValueError(field_name)
    number = int(numeric)
    if number < 0:
        raise ValueError(field_name)
    return number


def _finite_decimal(value: Any, field_name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(field_name) from error
    if not number.is_finite():
        raise ValueError(field_name)
    return number


def _usage_breakdown(response: Any) -> dict[str, int]:
    usage = _member(response, "usage")
    if usage is None:
        raise ValueError("usage")
    input_value = _member(usage, "input_tokens")
    output_value = _member(usage, "output_tokens")
    if input_value is None or output_value is None:
        raise ValueError("top-level usage")
    input_tokens = _as_nonnegative_int(input_value, "input_tokens")
    output_tokens = _as_nonnegative_int(output_value, "output_tokens")
    input_details = _member(usage, "input_tokens_details", {}) or {}
    output_details = _member(usage, "output_tokens_details", {}) or {}
    cached_tokens = _as_nonnegative_int(
        _member(input_details, "cached_tokens", 0) or 0,
        "cached_tokens",
    )
    cache_write_tokens = _as_nonnegative_int(
        _member(input_details, "cache_write_tokens", 0) or 0,
        "cache_write_tokens",
    )
    reasoning_tokens = _as_nonnegative_int(
        _member(output_details, "reasoning_tokens", 0) or 0,
        "reasoning_tokens",
    )
    total_tokens = _as_nonnegative_int(
        _member(usage, "total_tokens", input_tokens + output_tokens),
        "total_tokens",
    )
    if cached_tokens + cache_write_tokens > input_tokens:
        raise ValueError("input details exceed input tokens")
    if reasoning_tokens > output_tokens:
        raise ValueError("reasoning tokens exceed output tokens")
    if total_tokens != input_tokens + output_tokens:
        raise ValueError("total tokens are inconsistent")
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def _request_cost(model: str, usage: Mapping[str, int]) -> Decimal:
    pricing = MODEL_PRICING[model]
    input_multiplier = Decimal(1)
    output_multiplier = Decimal(1)
    if (
        pricing.long_context_threshold is not None
        and usage["input_tokens"] > pricing.long_context_threshold
    ):
        input_multiplier = pricing.long_input_multiplier
        output_multiplier = pricing.long_output_multiplier
    uncached = max(
        usage["input_tokens"]
        - usage["cached_input_tokens"]
        - usage["cache_write_tokens"],
        0,
    )
    million = Decimal(1_000_000)
    input_cost = (
        Decimal(uncached) * pricing.input
        + Decimal(usage["cached_input_tokens"]) * pricing.cached_input
        + Decimal(usage["cache_write_tokens"]) * pricing.cache_write
    ) * input_multiplier
    output_cost = Decimal(usage["output_tokens"]) * pricing.output * output_multiplier
    return (input_cost + output_cost) / million


def _conservative_usage(args: Sequence[Any], kwargs: Mapping[str, Any]) -> dict[str, int]:
    """Return a byte-bounded token estimate for an unaccounted request."""

    text_format = kwargs.get("text_format")
    schema_factory = getattr(text_format, "model_json_schema", None)
    schema = schema_factory() if callable(schema_factory) else str(text_format or "")
    payload = {
        "args": list(args),
        "input": kwargs.get("input"),
        "instructions": kwargs.get("instructions"),
        "text_format": schema,
    }
    try:
        serialized = json.dumps(
            payload,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except Exception:  # noqa: BLE001
        serialized = repr(payload)
    input_tokens = len(serialized.encode("utf-8"))
    configured_output = kwargs.get("max_output_tokens")
    output_tokens = (
        configured_output
        if isinstance(configured_output, int)
        and not isinstance(configured_output, bool)
        and configured_output > 0
        else CONSERVATIVE_MAX_OUTPUT_TOKENS
    )
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": output_tokens,
        "reasoning_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _filtered_organization_cost(
    cost_client: Any,
    *,
    execution_api_key_id: str,
    cost_window_start: int,
) -> Decimal:
    total = Decimal(0)
    page: str | None = None
    seen_pages: set[str] = set()
    while True:
        kwargs: dict[str, Any] = {
            "start_time": cost_window_start,
            "api_key_ids": [execution_api_key_id],
            "bucket_width": "1d",
            "limit": 180,
        }
        if page is not None:
            kwargs["page"] = page
        response = cost_client.admin.organization.usage.costs(**kwargs)
        for bucket in _member(response, "data", []) or []:
            for result in _member(bucket, "results", []) or []:
                amount = _member(result, "amount")
                currency = str(_member(amount, "currency", "")).lower()
                if currency != "usd":
                    raise ValueError("cost currency is not USD")
                total += _finite_decimal(_member(amount, "value"), "cost amount")
                if not total.is_finite():
                    raise ValueError("cost total")
        next_page = _member(response, "next_page")
        if not next_page:
            return total
        page = str(next_page)
        if page in seen_pages:
            raise ValueError("repeated cost cursor")
        seen_pages.add(page)


class _BudgetedResponses:
    def __init__(self, owner: BudgetedOpenAIClient) -> None:
        self._owner = owner

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner._parse(*args, **kwargs)


class BudgetedOpenAIClient:
    """Narrow Responses proxy with instance-local cost admission and accounting.

    Estimated-cost admission is the default. Its soft limit is evaluated after each
    accounted response, so the request that reaches the limit is already billable.
    """

    def __init__(
        self,
        *,
        api_key: str,
        max_budget_usd: Decimal | str,
        model_ids: Sequence[str],
        budget_source: Literal["estimated", "organization"] = "estimated",
        admin_api_key: str | None = None,
        execution_api_key_id: str | None = None,
        cost_window_start: int | None = None,
        client_factory: Callable[[str], Any] | None = None,
        cost_client_factory: Callable[[str], Any] | None = None,
        timeout: float = 120.0,
        max_retries: int = 0,
    ) -> None:
        try:
            unknown = sorted(set(model_ids) - MODEL_PRICING.keys())
            if unknown:
                raise UnknownModelPricingError(
                    "model pricing is not configured",
                    request_forwarded=False,
                )
            budget = _finite_decimal(max_budget_usd, "max_budget_usd")
            if budget <= 0:
                raise ValueError("max_budget_usd must be positive")
            if budget_source not in {"estimated", "organization"}:
                raise ValueError(
                    "budget_source must be 'estimated' or 'organization'"
                )
            if not isinstance(api_key, str) or not api_key:
                raise ValueError("api_key must be nonempty")
            if budget_source == "organization":
                if not isinstance(admin_api_key, str) or not admin_api_key:
                    raise ValueError(
                        "admin_api_key is required for organization mode"
                    )
                if (
                    not isinstance(execution_api_key_id, str)
                    or not execution_api_key_id
                ):
                    raise ValueError(
                        "execution_api_key_id is required for organization mode"
                    )
                if (
                    isinstance(cost_window_start, bool)
                    or not isinstance(cost_window_start, int)
                    or cost_window_start <= 0
                ):
                    raise ValueError(
                        "cost_window_start must be a positive integer in organization mode"
                    )

            response_client = None
            cost_client = None
            try:
                if client_factory is None or (
                    budget_source == "organization" and cost_client_factory is None
                ):
                    from openai import OpenAI

                response_client = (
                    client_factory(api_key)
                    if client_factory is not None
                    else OpenAI(
                        api_key=api_key,
                        timeout=timeout,
                        max_retries=max_retries,
                    )
                )
                if budget_source == "organization":
                    cost_client = (
                        cost_client_factory(admin_api_key)
                        if cost_client_factory is not None
                        else OpenAI(
                            admin_api_key=admin_api_key,
                            timeout=timeout,
                            max_retries=max_retries,
                        )
                    )
            except Exception as error:  # noqa: BLE001
                try:
                    for client in (response_client, cost_client):
                        try:
                            _close_client(client)
                        except Exception:  # noqa: BLE001, S110
                            pass
                finally:
                    response_client = None
                    cost_client = None
                raise ClientInitializationError(
                    f"OpenAI client construction failed ({type(error).__name__})"
                ) from None
        finally:
            api_key = ""
            admin_api_key = None

        self._response_client = response_client
        self._cost_client = cost_client
        self._execution_api_key_id = (
            execution_api_key_id if budget_source == "organization" else None
        )
        self._cost_window_start = (
            cost_window_start if budget_source == "organization" else None
        )
        self._max_budget_usd = budget
        self._budget_source = budget_source
        self._forward_requests = True
        self._closed = False
        self._forwarded_request_count = 0
        self._accounted_forwarded_request_count = 0
        self._unaccounted_forwarded_request_count = 0
        self._conservatively_accounted_request_count = 0
        self._totals = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
        self._estimated_cost_usd = Decimal(0)
        self._conservative_estimated_cost_usd = Decimal(0)
        self._last_request: dict[str, int] | None = None
        self._last_request_cost_usd = Decimal(0)
        self._last_request_accounted = False
        self._last_request_accounting_method = "none"
        self._organization_cost_usd_last_seen: Decimal | None = None
        self._cost_check_count = 0
        self._last_budget_check_error: dict[str, Any] | None = None
        self._accounting_error: dict[str, Any] | None = None
        self._accounting_warning_count = 0
        self._last_accounting_warning: dict[str, Any] | None = None
        self.responses = _BudgetedResponses(self)

    def _account_conservatively(
        self,
        model: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        error: BaseException,
    ) -> None:
        """Reserve an upper-bound estimated cost for one ambiguous request."""

        usage = _conservative_usage(args, kwargs)
        request_cost = _request_cost(model, usage)
        for name, value in usage.items():
            self._totals[name] += value
        self._estimated_cost_usd += request_cost
        self._conservative_estimated_cost_usd += request_cost
        self._accounted_forwarded_request_count += 1
        self._conservatively_accounted_request_count += 1
        self._last_request = dict(usage)
        self._last_request_cost_usd = request_cost
        self._last_request_accounted = True
        self._last_request_accounting_method = "conservative"
        self._accounting_error = None
        self._accounting_warning_count += 1
        self._last_accounting_warning = _safe_error(error)
        if (
            self._budget_source == "estimated"
            and self._estimated_cost_usd >= self._max_budget_usd
        ):
            self._forward_requests = False

    def _organization_cost(self) -> Decimal:
        self._cost_check_count += 1
        try:
            if self._cost_client is None:
                raise RuntimeError("cost client is unavailable")
            if self._execution_api_key_id is None or self._cost_window_start is None:
                raise RuntimeError("organization budget inputs are unavailable")
            total = _filtered_organization_cost(
                self._cost_client,
                execution_api_key_id=self._execution_api_key_id,
                cost_window_start=self._cost_window_start,
            )
            self._organization_cost_usd_last_seen = total
            self._last_budget_check_error = None
            return total
        except Exception as error:  # noqa: BLE001
            self._last_budget_check_error = _safe_error(error)
            raise BudgetCheckError(
                "filtered organization cost check failed",
                request_forwarded=False,
            ) from None

    def _parse(self, *args: Any, **kwargs: Any) -> Any:
        if not self._forward_requests:
            raise BudgetExceededError(
                "execution budget has been exhausted",
                request_forwarded=False,
            )
        if self._budget_source == "organization":
            organization_cost = self._organization_cost()
            if organization_cost >= self._max_budget_usd:
                self._forward_requests = False
                raise BudgetExceededError(
                    "execution budget has been exhausted",
                    request_forwarded=False,
                )
        model = kwargs.get("model")
        if not isinstance(model, str) or model not in MODEL_PRICING:
            raise UnknownModelPricingError(
                "model pricing is not configured",
                request_forwarded=False,
            )
        self._last_request = None
        self._last_request_cost_usd = Decimal(0)
        self._last_request_accounted = False
        self._last_request_accounting_method = "none"
        self._forwarded_request_count += 1
        try:
            response = self._response_client.responses.parse(*args, **kwargs)
        except Exception as error:
            safe_error = _safe_error(error)
            if (
                safe_error.get("status_code") == 400
                and safe_error.get("code") == "invalid_prompt"
            ):
                self._accounted_forwarded_request_count += 1
                self._last_request = {name: 0 for name in self._totals}
                self._last_request_accounted = True
                self._last_request_accounting_method = "rejected_zero"
                self._accounting_error = None
            elif _is_ambiguous_create_error(error):
                self._account_conservatively(model, args, kwargs, error)
            else:
                self._unaccounted_forwarded_request_count += 1
                self._accounting_error = safe_error
            raise
        try:
            usage = _usage_breakdown(response)
            request_cost = _request_cost(model, usage)
        except Exception as error:  # noqa: BLE001
            self._account_conservatively(model, args, kwargs, error)
            return response
        for name, value in usage.items():
            self._totals[name] += value
        self._estimated_cost_usd += request_cost
        self._accounted_forwarded_request_count += 1
        self._last_request = dict(usage)
        self._last_request_cost_usd = request_cost
        self._last_request_accounted = True
        self._last_request_accounting_method = "exact"
        self._accounting_error = None
        if (
            self._budget_source == "estimated"
            and self._estimated_cost_usd >= self._max_budget_usd
        ):
            self._forward_requests = False
        return response

    def usage_snapshot(self) -> dict[str, Any]:
        """Return the proxy's current secret-free accounting state."""

        return {
            "pricing_policy_version": PRICING_POLICY_VERSION,
            "budget_source": self._budget_source,
            "max_budget_usd": str(self._max_budget_usd),
            "execution_api_key_id": self._execution_api_key_id,
            "forwarded_request_count": self._forwarded_request_count,
            "accounted_forwarded_request_count": (
                self._accounted_forwarded_request_count
            ),
            "unaccounted_forwarded_request_count": (
                self._unaccounted_forwarded_request_count
            ),
            "conservatively_accounted_request_count": (
                self._conservatively_accounted_request_count
            ),
            **self._totals,
            "estimated_cost_usd": str(self._estimated_cost_usd),
            "conservative_estimated_cost_usd": str(
                self._conservative_estimated_cost_usd
            ),
            "last_request_tokens": (
                dict(self._last_request) if self._last_request is not None else None
            ),
            "last_request_cost_usd": str(self._last_request_cost_usd),
            "last_request_accounted": self._last_request_accounted,
            "last_request_accounting_method": self._last_request_accounting_method,
            "organization_cost_usd_last_seen": (
                str(self._organization_cost_usd_last_seen)
                if self._organization_cost_usd_last_seen is not None
                else None
            ),
            "cost_check_count": self._cost_check_count,
            "last_budget_check_error": self._last_budget_check_error,
            "forward_requests": self._forward_requests,
            "budget_exhausted": not self._forward_requests,
            "accounting_error": self._accounting_error,
            "accounting_warning_count": self._accounting_warning_count,
            "last_accounting_warning": self._last_accounting_warning,
            "estimated_cost_includes_conservative_estimates": bool(
                self._conservatively_accounted_request_count
            ),
            "estimated_cost_is_lower_bound": bool(
                self._unaccounted_forwarded_request_count
            ),
        }

    def close(self) -> None:
        """Close every constructed client once, attempting all on failure."""

        if self._closed:
            return
        self._closed = True
        failed = False
        for client in (self._response_client, self._cost_client):
            try:
                _close_client(client)
            except Exception:  # noqa: BLE001
                failed = True
        if failed:
            raise ClientCleanupError("one or more OpenAI clients failed to close")


def _is_ambiguous_create_error(error: BaseException) -> bool:
    status = getattr(error, "status_code", None)
    return (
        status is None
        or (isinstance(status, int) and status >= 500)
        or type(error).__name__ in {"APITimeoutError", "APIConnectionError"}
    )


def _recover_exact_service_account(
    admin_client: Any,
    *,
    project_id: str,
    name: str,
    max_pages: int,
) -> str | None:
    matches: list[str] = []
    page = admin_client.admin.organization.projects.service_accounts.list(
        project_id,
        limit=100,
    )
    fully_scanned = False
    for _ in range(max_pages):
        for item in _member(page, "data", []) or []:
            if _member(item, "name") == name and _member(item, "id"):
                matches.append(str(_member(item, "id")))
        has_next = getattr(page, "has_next_page", None)
        if not callable(has_next) or not has_next():
            fully_scanned = True
            break
        get_next = getattr(page, "get_next_page", None)
        if not callable(get_next):
            break
        page = get_next()
    if fully_scanned and len(matches) == 1:
        return matches[0]
    return None


@contextmanager
def temporary_execution_credential(
    *,
    admin_api_key: str,
    project_id: str,
    service_account_name: str,
    client_factory: Callable[[str], Any] | None = None,
    timeout: float = 120.0,
    max_retries: int = 0,
    max_recovery_pages: int = 10,
) -> Iterator[TemporaryExecutionCredential]:
    """Create and exactly revoke one project service-account API key."""

    response = None
    api_key = None
    raw_key = ""
    encoded_key = ""
    admin_client = None
    try:
        try:
            if client_factory is None:
                from openai import OpenAI

                admin_client = OpenAI(
                    admin_api_key=admin_api_key,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            else:
                admin_client = client_factory(admin_api_key)
        except Exception as error:  # noqa: BLE001
            raise ClientInitializationError(
                f"OpenAI Admin client construction failed ({type(error).__name__})"
            ) from None
    finally:
        admin_api_key = ""

    credential: TemporaryExecutionCredential | None = None
    service_account_id: str | None = None
    body_error: BaseException | None = None
    pending_invalid_response = False
    try:
        try:
            response = admin_client.admin.organization.projects.service_accounts.create(
                project_id,
                name=service_account_name,
            )
        except Exception as error:  # noqa: BLE001
            if not _is_ambiguous_create_error(error):
                raise TemporaryCredentialError(
                    "project service-account creation failed",
                    service_account_name=service_account_name,
                ) from None
            try:
                recovered_id = _recover_exact_service_account(
                    admin_client,
                    project_id=project_id,
                    name=service_account_name,
                    max_pages=max_recovery_pages,
                )
                if recovered_id is not None:
                    admin_client.admin.organization.projects.service_accounts.delete(
                        recovered_id,
                        project_id=project_id,
                    )
                    raise TemporaryCredentialError(
                        "ambiguous creation was recovered and cleaned up",
                        service_account_name=service_account_name,
                        cleanup_status="recovered_deleted",
                    ) from None
            except TemporaryCredentialError:
                raise
            except Exception:  # noqa: BLE001, S110
                pass
            raise TemporaryCredentialSecurityError(
                "ambiguous creation could not be recovered",
                service_account_name=service_account_name,
                possible_leak=True,
                cleanup_status="possible_leak",
            ) from None

        service_account_value = _member(response, "id")
        service_account_id = (
            service_account_value
            if isinstance(service_account_value, str) and service_account_value
            else ""
        )
        api_key = _member(response, "api_key")
        api_key_value = _member(api_key, "id")
        api_key_id = (
            api_key_value if isinstance(api_key_value, str) and api_key_value else ""
        )
        raw_key = _member(api_key, "value")
        try:
            if isinstance(api_key, Mapping):
                api_key["value"] = ""
            else:
                api_key.value = ""
        except Exception:  # noqa: BLE001, S110
            pass
        if not service_account_id or not api_key_id or not isinstance(raw_key, str):
            possible_leak = bool(
                not service_account_id and (api_key_id or isinstance(raw_key, str))
            )
            if service_account_id:
                pending_invalid_response = True
            raw_key = ""
            if not service_account_id:
                error_type = (
                    TemporaryCredentialSecurityError
                    if possible_leak
                    else TemporaryCredentialError
                )
                raise error_type(
                    "service-account response omitted required credential fields",
                    service_account_name=service_account_name,
                    possible_leak=possible_leak,
                    cleanup_status=(
                        "invalid_response_possible_leak"
                        if possible_leak
                        else "not_created"
                    ),
                )
        else:
            encoded_key = base64.b64encode(raw_key.encode("utf-8")).decode("ascii")
            raw_key = ""
            credential = TemporaryExecutionCredential(
                project_id=project_id,
                service_account_id=service_account_id,
                api_key_id=api_key_id,
                service_account_name=service_account_name,
                encoded_key=encoded_key,
            )
            try:
                yield credential
            except BaseException as error:
                body_error = error
                raise
    finally:
        if credential is not None:
            credential.clear_secret()
        try:
            if isinstance(api_key, dict):
                api_key.clear()
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            if isinstance(response, dict):
                response.clear()
        except Exception:  # noqa: BLE001, S110
            pass
        response = None
        api_key = None
        raw_key = ""
        encoded_key = ""
        deletion_failed = False
        deletion_error: BaseException | None = None
        if service_account_id:
            try:
                admin_client.admin.organization.projects.service_accounts.delete(
                    service_account_id,
                    project_id=project_id,
                )
                if credential is not None:
                    credential.cleanup_status = "deleted"
            except BaseException as error:  # noqa: BLE001
                deletion_failed = True
                deletion_error = error
                if credential is not None:
                    credential.cleanup_status = "failed"
                    credential.possible_leak = True
        close_error: BaseException | None = None
        try:
            try:
                _close_client(admin_client)
            except BaseException as error:  # noqa: BLE001
                close_error = error
                if credential is not None and credential.cleanup_status == "deleted":
                    credential.cleanup_status = "deleted_client_close_failed"
        finally:
            admin_client = None

        cleanup_failures = {
            stage: _safe_error(error)
            for stage, error in (
                ("body", body_error),
                ("deletion", deletion_error),
                ("close", close_error),
            )
            if error is not None
        }

        if pending_invalid_response:
            if deletion_failed:
                invalid_response_error = TemporaryCredentialSecurityError(
                    "service-account response omitted required credential fields",
                    service_account_name=service_account_name,
                    possible_leak=True,
                    cleanup_status="invalid_response_delete_failed",
                )
                invalid_response_error.cleanup_failures = cleanup_failures
                raise invalid_response_error from (
                    body_error if body_error is not None else deletion_error
                )
            invalid_response_error = TemporaryCredentialError(
                "service-account response omitted required credential fields",
                service_account_name=service_account_name,
                cleanup_status="invalid_response_deleted",
            )
            invalid_response_error.cleanup_failures = cleanup_failures
            raise invalid_response_error from body_error

        if credential is not None and credential.possible_leak:
            security_error = TemporaryCredentialSecurityError(
                "project service-account deletion failed",
                service_account_name=service_account_name,
                possible_leak=True,
                cleanup_status=credential.cleanup_status,
            )
            security_error.cleanup_failures = cleanup_failures
            if body_error is not None:
                raise security_error from body_error
            raise security_error from deletion_error

        if (
            close_error is not None
            and not isinstance(close_error, Exception)
            and body_error is None
        ):
            raise close_error


def validate_temporary_model_access(
    *,
    api_key: str,
    model_ids: Sequence[str],
    client_factory: Callable[[str], Any] | None = None,
    timeout: float = 120.0,
    max_retries: int = 0,
) -> list[str]:
    """Require a temporary key to list every configured execution model."""

    client = None
    try:
        if client_factory is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        else:
            client = client_factory(api_key)
        visible = {
            str(_member(model, "id"))
            for model in client.models.list()
            if _member(model, "id")
        }
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            f"temporary-key model preflight failed ({type(error).__name__})"
        ) from None
    finally:
        api_key = ""
        try:
            if client is not None:
                try:
                    _close_client(client)
                except Exception:  # noqa: BLE001, S110
                    pass
        finally:
            client = None
    missing = sorted(set(model_ids) - visible)
    if missing:
        raise RuntimeError(f"temporary key cannot access configured models: {missing}")
    return sorted(visible)


def validate_admin_project_access(
    *,
    admin_api_key: str,
    project_id: str,
    client_factory: Callable[[str], Any] | None = None,
    timeout: float = 120.0,
    max_retries: int = 0,
) -> None:
    """Verify that an Admin key can access the configured OpenAI project."""

    client = None
    try:
        if client_factory is None:
            from openai import OpenAI

            client = OpenAI(
                admin_api_key=admin_api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            client = client_factory(admin_api_key)
        client.admin.organization.projects.service_accounts.list(
            project_id,
            limit=1,
        )
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            f"OpenAI Admin project preflight failed ({type(error).__name__})"
        ) from None
    finally:
        admin_api_key = ""
        try:
            if client is not None:
                try:
                    _close_client(client)
                except Exception:  # noqa: BLE001, S110
                    pass
        finally:
            client = None


def validate_filtered_cost_access(
    *,
    admin_api_key: str,
    execution_api_key_id: str,
    cost_window_start: int,
    client_factory: Callable[[str], Any] | None = None,
    timeout: float = 120.0,
    max_retries: int = 0,
) -> str:
    """Verify filtered Costs access and return the initial USD observation."""

    client = None
    try:
        if client_factory is None:
            from openai import OpenAI

            client = OpenAI(
                admin_api_key=admin_api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            client = client_factory(admin_api_key)
        cost = _filtered_organization_cost(
            client,
            execution_api_key_id=execution_api_key_id,
            cost_window_start=cost_window_start,
        )
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            f"filtered OpenAI Costs preflight failed ({type(error).__name__})"
        ) from None
    finally:
        admin_api_key = ""
        try:
            if client is not None:
                try:
                    _close_client(client)
                except Exception:  # noqa: BLE001, S110
                    pass
        finally:
            client = None
    return str(cost)


__all__ = [
    "MODEL_PRICING",
    "PRICING_POLICY_VERSION",
    "BudgetCheckError",
    "BudgetExceededError",
    "BudgetedOpenAIClient",
    "ClientCleanupError",
    "ClientInitializationError",
    "ModelPricing",
    "ProxyRequestError",
    "TemporaryCredentialError",
    "TemporaryCredentialSecurityError",
    "TemporaryExecutionCredential",
    "UnknownModelPricingError",
    "UsageAccountingError",
    "temporary_execution_credential",
    "validate_admin_project_access",
    "validate_filtered_cost_access",
    "validate_temporary_model_access",
]
