from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest
from mha_exp_common.openai_budget import (
    BudgetCheckError,
    BudgetedOpenAIClient,
    BudgetExceededError,
    ClientCleanupError,
    ClientInitializationError,
    TemporaryCredentialError,
    TemporaryCredentialSecurityError,
    UnknownModelPricingError,
    temporary_execution_credential,
    validate_admin_project_access,
    validate_filtered_cost_access,
    validate_temporary_model_access,
)


def _repository_traceback_locals(error: BaseException) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    traceback = error.__traceback__
    while traceback is not None:
        filename = traceback.tb_frame.f_code.co_filename.replace("\\", "/")
        if "/mha_exp_common/src/" in filename:
            frames.append(dict(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    assert frames
    return frames


class FakeResponses:
    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeResponseClient:
    def __init__(self, response: object | None = None) -> None:
        self.responses = FakeResponses(response)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeCostClient:
    def __init__(self, values: list[str] | None = None, error: Exception | None = None):
        self.values = values or ["0"]
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.closed = False
        self.admin = SimpleNamespace(
            organization=SimpleNamespace(
                usage=SimpleNamespace(costs=self.costs),
            )
        )

    def costs(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        value = self.values[len(self.calls) - 1]
        result = SimpleNamespace(amount=SimpleNamespace(value=value, currency="usd"))
        return SimpleNamespace(
            data=[SimpleNamespace(results=[result])],
            next_page=None,
        )

    def close(self) -> None:
        self.closed = True


def response(
    *,
    input_tokens: int = 1000,
    cached_tokens: int = 100,
    cache_write_tokens: int = 50,
    output_tokens: int = 200,
    reasoning_tokens: int = 20,
) -> object:
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        input_tokens_details=SimpleNamespace(
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
        ),
        output_tokens=output_tokens,
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        total_tokens=input_tokens + output_tokens,
    )
    return SimpleNamespace(id="resp", usage=usage)


def proxy(
    response_value: object,
    cost_client: FakeCostClient,
    *,
    budget: str = "4.00",
    budget_source: str = "estimated",
) -> tuple[BudgetedOpenAIClient, FakeResponseClient]:
    response_client = FakeResponseClient(response_value)
    organization_kwargs = (
        {
            "admin_api_key": "admin-secret",
            "execution_api_key_id": "key-1",
            "cost_window_start": 123,
        }
        if budget_source == "organization"
        else {}
    )
    client = BudgetedOpenAIClient(
        api_key="response-secret",
        max_budget_usd=budget,
        model_ids=["gpt-5.4-nano"],
        budget_source=budget_source,
        client_factory=lambda _: response_client,
        cost_client_factory=lambda _: cost_client,
        **organization_kwargs,
    )
    return client, response_client


def test_parse_forwards_unchanged_and_accounts_detailed_usage() -> None:
    cost_client = FakeCostClient()
    client, response_client = proxy(response(), cost_client)

    result = client.responses.parse(model="gpt-5.4-nano", input="hello")

    assert result.id == "resp"
    assert response_client.responses.calls == [
        {"model": "gpt-5.4-nano", "input": "hello"}
    ]
    assert cost_client.calls == []
    snapshot = client.usage_snapshot()
    assert snapshot["budget_source"] == "estimated"
    assert snapshot["cached_input_tokens"] == 100
    assert snapshot["cache_write_tokens"] == 50
    assert snapshot["reasoning_tokens"] == 20
    assert snapshot["estimated_cost_usd"] == "0.000432"
    assert snapshot["execution_api_key_id"] is None
    assert snapshot["organization_cost_usd_last_seen"] is None
    assert snapshot["cost_check_count"] == 0
    assert snapshot["last_budget_check_error"] is None


def test_budget_gate_switches_once_and_stops_both_clients() -> None:
    cost_client = FakeCostClient()
    client, response_client = proxy(response(), cost_client, budget="0.0004")

    result = client.responses.parse(model="gpt-5.4-nano")

    with pytest.raises(BudgetExceededError) as first:
        client.responses.parse(model="gpt-5.4-nano")
    with pytest.raises(BudgetExceededError) as second:
        client.responses.parse(model="gpt-5.4-nano")

    assert result.id == "resp"
    assert first.value.request_forwarded is False
    assert second.value.request_forwarded is False
    assert cost_client.calls == []
    assert len(response_client.responses.calls) == 1
    assert client.usage_snapshot()["forwarded_request_count"] == 1
    assert client.usage_snapshot()["budget_exhausted"] is True


def test_organization_budget_source_remains_available() -> None:
    cost_client = FakeCostClient(["4.00"])
    client, response_client = proxy(
        response(), cost_client, budget_source="organization"
    )

    with pytest.raises(BudgetExceededError) as first:
        client.responses.parse(model="gpt-5.4-nano")
    with pytest.raises(BudgetExceededError) as second:
        client.responses.parse(model="gpt-5.4-nano")

    assert first.value.request_forwarded is False
    assert second.value.request_forwarded is False
    assert len(cost_client.calls) == 1
    assert not response_client.responses.calls
    assert client.usage_snapshot()["budget_source"] == "organization"


def test_estimated_default_does_not_construct_cost_client() -> None:
    response_client = FakeResponseClient(response())

    def fail_cost_client(_: str) -> object:
        raise AssertionError("estimated mode must not construct a Costs client")

    client = BudgetedOpenAIClient(
        api_key="response-secret",
        max_budget_usd="1",
        model_ids=["gpt-5.4-nano"],
        client_factory=lambda _: response_client,
        cost_client_factory=fail_cost_client,
    )

    client.responses.parse(model="gpt-5.4-nano")

    assert client.usage_snapshot()["budget_source"] == "estimated"


@pytest.mark.parametrize(
    "organization_kwargs",
    [
        {"execution_api_key_id": "key", "cost_window_start": 1},
        {"admin_api_key": "admin", "cost_window_start": 1},
        {"admin_api_key": "admin", "execution_api_key_id": "key"},
    ],
)
def test_organization_inputs_are_required_before_client_construction(
    organization_kwargs: dict[str, object],
) -> None:
    calls: list[str] = []

    with pytest.raises(ValueError):
        BudgetedOpenAIClient(
            api_key="response-secret",
            max_budget_usd="1",
            model_ids=["gpt-5.4-nano"],
            budget_source="organization",
            client_factory=lambda _: calls.append("response"),
            cost_client_factory=lambda _: calls.append("cost"),
            **organization_kwargs,
        )

    assert calls == []


def test_cost_failure_is_retryable_without_disabling_gate() -> None:
    cost_client = FakeCostClient(error=TimeoutError("secret detail"))
    client, response_client = proxy(
        response(), cost_client, budget_source="organization"
    )

    with pytest.raises(BudgetCheckError) as caught:
        client.responses.parse(model="gpt-5.4-nano")

    assert caught.value.request_forwarded is False
    assert "secret detail" not in str(caught.value)
    assert client.usage_snapshot()["forward_requests"] is True
    assert not response_client.responses.calls


def test_filtered_cost_preflight_uses_execution_key() -> None:
    cost_client = FakeCostClient(["0.125"])

    observed = validate_filtered_cost_access(
        admin_api_key="admin-secret",
        execution_api_key_id="key-1",
        cost_window_start=123,
        client_factory=lambda _: cost_client,
    )

    assert observed == "0.125"
    assert cost_client.calls == [
        {
            "start_time": 123,
            "api_key_ids": ["key-1"],
            "bucket_width": "1d",
            "limit": 180,
        }
    ]


def test_unknown_model_and_malformed_usage_use_conservative_accounting() -> None:
    with pytest.raises(UnknownModelPricingError):
        BudgetedOpenAIClient(
            api_key="x",
            admin_api_key="y",
            execution_api_key_id="key",
            cost_window_start=1,
            max_budget_usd="1",
            model_ids=["unknown"],
            client_factory=lambda _: FakeResponseClient(),
            cost_client_factory=lambda _: FakeCostClient(),
        )

    malformed = SimpleNamespace(id="resp_bad", usage=None)
    client, _ = proxy(malformed, FakeCostClient())
    assert (
        client.responses.parse(
            model="gpt-5.4-nano",
            input="hello",
            max_output_tokens=200,
        )
        is malformed
    )
    snapshot = client.usage_snapshot()
    assert snapshot["accounted_forwarded_request_count"] == 1
    assert snapshot["unaccounted_forwarded_request_count"] == 0
    assert snapshot["conservatively_accounted_request_count"] == 1
    assert snapshot["last_request_tokens"]["output_tokens"] == 200
    assert Decimal(snapshot["last_request_cost_usd"]) > 0
    assert snapshot["last_request_accounted"] is True
    assert snapshot["last_request_accounting_method"] == "conservative"
    assert snapshot["accounting_warning_count"] == 1
    assert snapshot["estimated_cost_includes_conservative_estimates"] is True
    assert snapshot["estimated_cost_is_lower_bound"] is False


def test_ambiguous_parse_error_is_conservatively_accounted_and_rethrown() -> None:
    response_client = FakeResponseClient(response())
    client = BudgetedOpenAIClient(
        api_key="x",
        max_budget_usd="1",
        model_ids=["gpt-5.4-nano"],
        client_factory=lambda _: response_client,
    )
    client.responses.parse(model="gpt-5.4-nano")
    response_client.responses.error = TimeoutError("secret detail")

    with pytest.raises(TimeoutError):
        client.responses.parse(
            model="gpt-5.4-nano",
            input="retry this request",
            max_output_tokens=300,
        )

    snapshot = client.usage_snapshot()
    assert snapshot["forwarded_request_count"] == 2
    assert snapshot["accounted_forwarded_request_count"] == 2
    assert snapshot["unaccounted_forwarded_request_count"] == 0
    assert snapshot["conservatively_accounted_request_count"] == 1
    assert snapshot["last_request_tokens"]["output_tokens"] == 300
    assert snapshot["last_request_accounted"] is True
    assert snapshot["last_request_accounting_method"] == "conservative"
    assert snapshot["estimated_cost_is_lower_bound"] is False
    assert snapshot["accounting_error"] is None
    assert snapshot["last_accounting_warning"]["type"] == "TimeoutError"


def test_rejected_invalid_prompt_is_accounted_as_zero_usage() -> None:
    class InvalidPromptError(Exception):
        status_code = 400
        code = "invalid_prompt"

    client, response_client = proxy(response(), FakeCostClient())
    response_client.responses.error = InvalidPromptError("sanitized by caller")

    with pytest.raises(InvalidPromptError):
        client.responses.parse(model="gpt-5.4-nano")

    snapshot = client.usage_snapshot()
    assert snapshot["forwarded_request_count"] == 1
    assert snapshot["accounted_forwarded_request_count"] == 1
    assert snapshot["unaccounted_forwarded_request_count"] == 0
    assert snapshot["last_request_tokens"] == {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    assert snapshot["last_request_cost_usd"] == "0"
    assert snapshot["last_request_accounted"] is True
    assert snapshot["estimated_cost_is_lower_bound"] is False


def test_luna_long_context_multiplier_applies_to_whole_request() -> None:
    response_client = FakeResponseClient(
        response(
            input_tokens=273_000,
            cached_tokens=0,
            cache_write_tokens=0,
            output_tokens=1_000,
            reasoning_tokens=0,
        )
    )
    client = BudgetedOpenAIClient(
        api_key="x",
        max_budget_usd="4",
        model_ids=["gpt-5.6-luna"],
        client_factory=lambda _: response_client,
        cost_client_factory=lambda _: FakeCostClient(),
    )

    client.responses.parse(model="gpt-5.6-luna")

    assert client.usage_snapshot()["estimated_cost_usd"] == "0.111"


def test_close_attempts_both_clients_and_is_idempotent() -> None:
    class BrokenResponseClient(FakeResponseClient):
        def close(self) -> None:
            self.closed = True
            raise RuntimeError("secret")

    response_client = BrokenResponseClient(response())
    cost_client = FakeCostClient()
    client = BudgetedOpenAIClient(
        api_key="x",
        admin_api_key="y",
        execution_api_key_id="key",
        cost_window_start=1,
        max_budget_usd=Decimal(1),
        model_ids=["gpt-5.4-nano"],
        budget_source="organization",
        client_factory=lambda _: response_client,
        cost_client_factory=lambda _: cost_client,
    )
    with pytest.raises(ClientCleanupError):
        client.close()
    client.close()
    assert response_client.closed and cost_client.closed


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "not-number"])
def test_budget_rejects_nonfinite_and_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="max_budget_usd"):
        proxy(response(), FakeCostClient(), budget=value)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "not-number"])
def test_invalid_cost_values_are_nonforwarded_budget_failures(value: str) -> None:
    client, response_client = proxy(
        response(),
        FakeCostClient([value]),
        budget_source="organization",
    )

    with pytest.raises(BudgetCheckError) as caught:
        client.responses.parse(model="gpt-5.4-nano")

    assert caught.value.request_forwarded is False
    assert not response_client.responses.calls
    assert client.usage_snapshot()["forwarded_request_count"] == 0


def test_fractional_token_usage_uses_conservative_accounting() -> None:
    malformed = response(input_tokens=1000)
    malformed.usage.input_tokens = 1.5
    client, response_client = proxy(malformed, FakeCostClient())

    assert client.responses.parse(model="gpt-5.4-nano") is malformed
    assert len(response_client.responses.calls) == 1
    snapshot = client.usage_snapshot()
    assert snapshot["conservatively_accounted_request_count"] == 1
    assert snapshot["last_request_accounting_method"] == "conservative"


def test_conservative_accounting_can_stop_future_requests() -> None:
    client, response_client = proxy(
        response(),
        FakeCostClient(),
        budget="0.0001",
    )
    response_client.responses.error = TimeoutError("ambiguous delivery")

    with pytest.raises(TimeoutError):
        client.responses.parse(
            model="gpt-5.4-nano",
            input="hello",
            max_output_tokens=1000,
        )

    snapshot = client.usage_snapshot()
    assert snapshot["budget_exhausted"] is True
    assert snapshot["forward_requests"] is False
    assert snapshot["conservatively_accounted_request_count"] == 1
    with pytest.raises(BudgetExceededError) as caught:
        client.responses.parse(model="gpt-5.4-nano", input="blocked")
    assert caught.value.request_forwarded is False
    assert len(response_client.responses.calls) == 1


def test_partial_client_construction_closes_first_client() -> None:
    response_client = FakeResponseClient()

    def fail_cost_client(_: str) -> object:
        raise RuntimeError("secret construction detail")

    with pytest.raises(ClientInitializationError) as caught:
        BudgetedOpenAIClient(
            api_key="response-secret",
            admin_api_key="admin-secret",
            execution_api_key_id="key",
            cost_window_start=1,
            max_budget_usd="1",
            model_ids=["gpt-5.4-nano"],
            budget_source="organization",
            client_factory=lambda _: response_client,
            cost_client_factory=fail_cost_client,
        )

    assert response_client.closed
    assert "secret construction detail" not in str(caught.value)


def test_client_construction_failure_clears_keys_from_repository_traceback() -> None:
    configured = "configured-client-construction-marker"
    admin = "admin-client-construction-marker"

    def fail_response_client(_: str) -> object:
        raise RuntimeError("construction failed")

    with pytest.raises(ClientInitializationError) as caught:
        BudgetedOpenAIClient(
            api_key=configured,
            admin_api_key=admin,
            execution_api_key_id="key",
            cost_window_start=1,
            max_budget_usd="1",
            model_ids=["gpt-5.4-nano"],
            budget_source="organization",
            client_factory=fail_response_client,
            cost_client_factory=lambda _: FakeCostClient(),
        )

    frames = _repository_traceback_locals(caught.value)
    assert configured not in repr(frames)
    assert admin not in repr(frames)
    assert all(frame.get("api_key") == "" for frame in frames)
    assert all(frame.get("admin_api_key") is None for frame in frames)


def test_filtered_cost_access_sums_pages_and_results() -> None:
    calls: list[dict[str, object]] = []

    def costs(**kwargs: object) -> object:
        calls.append(kwargs)
        values, next_page = (
            (["0.10", "0.20"], "cursor-2") if "page" not in kwargs else (["0.03"], None)
        )
        results = [
            SimpleNamespace(amount=SimpleNamespace(value=value, currency="usd"))
            for value in values
        ]
        return SimpleNamespace(
            data=[SimpleNamespace(results=results)],
            next_page=next_page,
        )

    client = FakeCostClient()
    client.admin.organization.usage.costs = costs

    observed = validate_filtered_cost_access(
        admin_api_key="admin",
        execution_api_key_id="key",
        cost_window_start=1,
        client_factory=lambda _: client,
    )

    assert observed == "0.33"
    assert calls[1]["page"] == "cursor-2"


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(data=[], next_page="same"),
        SimpleNamespace(
            data=[
                SimpleNamespace(
                    results=[
                        SimpleNamespace(
                            amount=SimpleNamespace(value="1", currency="eur")
                        )
                    ]
                )
            ],
            next_page=None,
        ),
    ],
)
def test_filtered_cost_access_rejects_repeated_cursor_and_non_usd(
    response: object,
) -> None:
    client = FakeCostClient()
    client.admin.organization.usage.costs = lambda **kwargs: response

    with pytest.raises(RuntimeError, match="Costs preflight failed"):
        validate_filtered_cost_access(
            admin_api_key="admin",
            execution_api_key_id="key",
            cost_window_start=1,
            client_factory=lambda _: client,
        )


class FakeServiceAccounts:
    def __init__(
        self,
        *,
        create_error: Exception | None = None,
        delete_error: Exception | None = None,
        list_data: list[object] | None = None,
        create_response: object | None = None,
    ) -> None:
        self.create_error = create_error
        self.delete_error = delete_error
        self.deleted: list[str] = []
        self.list_calls = 0
        self.created_key = SimpleNamespace(id="api-key-1", value="temporary-secret")
        self.list_data = list_data or []
        self.create_response = create_response

    def create(self, project_id: str, *, name: str) -> object:
        if self.create_error is not None:
            raise self.create_error
        return self.create_response or SimpleNamespace(
            id="service-1", api_key=self.created_key
        )

    def delete(self, service_id: str, *, project_id: str) -> object:
        self.deleted.append(service_id)
        if self.delete_error is not None:
            raise self.delete_error
        return SimpleNamespace(deleted=True)

    def list(self, project_id: str, *, limit: int) -> object:
        self.list_calls += 1
        return SimpleNamespace(
            data=self.list_data,
            has_next_page=lambda: False,
        )


def admin_client(service_accounts: FakeServiceAccounts) -> object:
    return SimpleNamespace(
        admin=SimpleNamespace(
            organization=SimpleNamespace(
                projects=SimpleNamespace(service_accounts=service_accounts)
            )
        ),
        close=lambda: None,
    )


def test_temporary_credential_clears_secret_and_deletes_exact_id() -> None:
    accounts = FakeServiceAccounts()
    holder = None
    with temporary_execution_credential(
        admin_api_key="admin-secret",
        project_id="project-1",
        service_account_name="exact-name",
        client_factory=lambda _: admin_client(accounts),
    ) as credential:
        holder = credential
        assert "temporary-secret" not in repr(credential)
        assert credential.api_key_id == "api-key-1"
        assert accounts.created_key.value == ""
    assert holder is not None
    assert holder.encoded_key == ""
    assert holder.cleanup_status == "deleted"
    assert accounts.deleted == ["service-1"]


def test_ambiguous_creation_without_exact_match_reports_possible_leak() -> None:
    accounts = FakeServiceAccounts(create_error=TimeoutError("secret"))
    with (
        pytest.raises(TemporaryCredentialError) as caught,
        temporary_execution_credential(
            admin_api_key="admin",
            project_id="project",
            service_account_name="unique-name",
            client_factory=lambda _: admin_client(accounts),
        ),
    ):
        pytest.fail("must not yield")
    assert caught.value.possible_leak is True
    assert "secret" not in str(caught.value)
    assert not accounts.deleted


def test_ambiguous_creation_recovers_one_exact_name_and_deletes_it() -> None:
    accounts = FakeServiceAccounts(
        create_error=TimeoutError("secret"),
        list_data=[SimpleNamespace(id="recovered-1", name="unique-name")],
    )

    with (
        pytest.raises(TemporaryCredentialError) as caught,
        temporary_execution_credential(
            admin_api_key="admin",
            project_id="project",
            service_account_name="unique-name",
            client_factory=lambda _: admin_client(accounts),
        ),
    ):
        pytest.fail("must not yield")

    assert caught.value.cleanup_status == "recovered_deleted"
    assert caught.value.possible_leak is False
    assert accounts.deleted == ["recovered-1"]


def test_ambiguous_creation_does_not_guess_between_exact_matches() -> None:
    accounts = FakeServiceAccounts(
        create_error=TimeoutError("secret"),
        list_data=[
            SimpleNamespace(id="match-1", name="unique-name"),
            SimpleNamespace(id="match-2", name="unique-name"),
        ],
    )

    with (
        pytest.raises(TemporaryCredentialError) as caught,
        temporary_execution_credential(
            admin_api_key="admin",
            project_id="project",
            service_account_name="unique-name",
            client_factory=lambda _: admin_client(accounts),
        ),
    ):
        pytest.fail("must not yield")

    assert caught.value.possible_leak is True
    assert not accounts.deleted


def test_deterministic_creation_failure_does_not_list_or_delete() -> None:
    class BadRequestError(Exception):
        status_code = 400

    accounts = FakeServiceAccounts(create_error=BadRequestError("secret"))

    with (
        pytest.raises(TemporaryCredentialError) as caught,
        temporary_execution_credential(
            admin_api_key="admin",
            project_id="project",
            service_account_name="unique-name",
            client_factory=lambda _: admin_client(accounts),
        ),
    ):
        pytest.fail("must not yield")

    assert accounts.list_calls == 0
    assert not accounts.deleted
    assert "secret" not in str(caught.value)


def test_missing_api_key_fields_delete_known_service_account() -> None:
    accounts = FakeServiceAccounts(
        create_response=SimpleNamespace(id="service-1", api_key=None)
    )

    with (
        pytest.raises(TemporaryCredentialError) as caught,
        temporary_execution_credential(
            admin_api_key="admin",
            project_id="project",
            service_account_name="unique-name",
            client_factory=lambda _: admin_client(accounts),
        ),
    ):
        pytest.fail("must not yield")

    assert caught.value.cleanup_status == "invalid_response_deleted"
    assert accounts.deleted == ["service-1"]


def test_missing_service_account_id_reports_possible_leak_and_clears_key() -> None:
    accounts = FakeServiceAccounts(
        create_response=SimpleNamespace(id=None, api_key=None)
    )
    accounts.create_response.api_key = accounts.created_key

    with (
        pytest.raises(TemporaryCredentialError) as caught,
        temporary_execution_credential(
            admin_api_key="admin",
            project_id="project",
            service_account_name="unique-name",
            client_factory=lambda _: admin_client(accounts),
        ),
    ):
        pytest.fail("must not yield")

    assert caught.value.cleanup_status == "invalid_response_possible_leak"
    assert caught.value.possible_leak is True
    assert accounts.created_key.value == ""
    assert not accounts.deleted


def test_deletion_failure_after_successful_body_fails_with_possible_leak() -> None:
    accounts = FakeServiceAccounts(delete_error=RuntimeError("secret delete detail"))
    holder = None

    with (
        pytest.raises(TemporaryCredentialSecurityError) as caught,
        temporary_execution_credential(
            admin_api_key="admin",
            project_id="project",
            service_account_name="unique-name",
            client_factory=lambda _: admin_client(accounts),
        ) as credential,
    ):
        holder = credential

    assert holder is not None
    assert holder.cleanup_status == "failed"
    assert holder.possible_leak is True
    assert holder.encoded_key == ""
    assert caught.value.possible_leak is True
    assert "secret delete detail" not in str(caught.value)


def test_temporary_credential_cleanup_clears_secrets_from_repository_traceback() -> None:
    configured = "temporary-created-key-marker"
    admin = "temporary-admin-key-marker"
    accounts = FakeServiceAccounts(delete_error=RuntimeError("delete failed"))
    accounts.created_key.value = configured

    with (
        pytest.raises(TemporaryCredentialSecurityError) as caught,
        temporary_execution_credential(
            admin_api_key=admin,
            project_id="project",
            service_account_name="unique-name",
            client_factory=lambda _: admin_client(accounts),
        ),
    ):
        pass

    frames = _repository_traceback_locals(caught.value)
    assert configured not in repr(frames)
    assert admin not in repr(frames)
    assert accounts.created_key.value == ""
    frame = next(
        item for item in frames if "service_account_name" in item
    )
    assert frame["admin_api_key"] == ""
    assert frame["raw_key"] == ""
    assert frame["encoded_key"] == ""
    assert frame["api_key"] is None
    assert frame["response"] is None
    assert frame["admin_client"] is None


@pytest.mark.parametrize(
    "body_error",
    [None, ValueError("body failed")],
    ids=["without-body-error", "with-body-error"],
)
def test_revocation_failure_remains_primary_when_admin_close_is_interrupted(
    body_error: BaseException | None,
) -> None:
    configured = "double-failure-temporary-key-marker"
    admin = "double-failure-admin-key-marker"
    deletion_error = RuntimeError("deletion failed")
    close_error = KeyboardInterrupt("close interrupted")
    accounts = FakeServiceAccounts(delete_error=deletion_error)
    accounts.created_key.value = configured
    client = admin_client(accounts)

    def fail_close() -> None:
        raise close_error

    client.close = fail_close
    holder = None
    with (
        pytest.raises(TemporaryCredentialSecurityError) as caught,
        temporary_execution_credential(
            admin_api_key=admin,
            project_id="project",
            service_account_name="unique-name",
            client_factory=lambda _: client,
        ) as credential,
    ):
        holder = credential
        if body_error is not None:
            raise body_error

    assert holder is not None
    assert holder.cleanup_status == "failed"
    assert holder.possible_leak is True
    assert holder.encoded_key == ""
    assert accounts.created_key.value == ""
    assert caught.value.__cause__ is (body_error or deletion_error)
    expected_failures = {
        "deletion": {"type": "RuntimeError"},
        "close": {"type": "KeyboardInterrupt"},
    }
    if body_error is not None:
        expected_failures["body"] = {"type": "ValueError"}
    assert caught.value.cleanup_failures == expected_failures
    assert configured not in repr(caught.value.cleanup_failures)
    assert admin not in repr(caught.value.cleanup_failures)
    frames = _repository_traceback_locals(caught.value)
    assert configured not in repr(frames)
    assert admin not in repr(frames)


def test_temporary_admin_client_failure_clears_key_from_repository_traceback() -> None:
    admin = "temporary-admin-construction-marker"

    def fail_client(_: str) -> object:
        raise RuntimeError("construction failed")

    with (
        pytest.raises(ClientInitializationError) as caught,
        temporary_execution_credential(
            admin_api_key=admin,
            project_id="project",
            service_account_name="unique-name",
            client_factory=fail_client,
        ),
    ):
        pass

    frames = _repository_traceback_locals(caught.value)
    assert admin not in repr(frames)
    assert all(frame.get("admin_api_key") == "" for frame in frames)


def test_revocation_failure_overrides_base_exception_body() -> None:
    class StopExecution(BaseException):
        pass

    accounts = FakeServiceAccounts(delete_error=RuntimeError("secret detail"))
    body_error = StopExecution("stop")

    with (
        pytest.raises(TemporaryCredentialSecurityError) as caught,
        temporary_execution_credential(
            admin_api_key="admin",
            project_id="project",
            service_account_name="unique-name",
            client_factory=lambda _: admin_client(accounts),
        ),
    ):
        raise body_error

    assert caught.value.__cause__ is body_error


def test_deletion_failure_overrides_and_chains_existing_body_error() -> None:
    accounts = FakeServiceAccounts(delete_error=RuntimeError("secret delete detail"))
    holder = None

    with (
        pytest.raises(TemporaryCredentialSecurityError) as caught,
        temporary_execution_credential(
            admin_api_key="admin",
            project_id="project",
            service_account_name="unique-name",
            client_factory=lambda _: admin_client(accounts),
        ) as credential,
    ):
        holder = credential
        raise ValueError("body failed")

    assert holder is not None
    assert holder.cleanup_status == "failed"
    assert holder.possible_leak is True
    assert holder.encoded_key == ""
    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value.__cause__) == "body failed"


def test_temporary_model_access_requires_configured_models_and_closes() -> None:
    client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: [
                SimpleNamespace(id="gpt-5.4-nano"),
                SimpleNamespace(id="unrelated-model"),
            ]
        ),
        closed=False,
    )

    def close() -> None:
        client.closed = True

    client.close = close
    visible = validate_temporary_model_access(
        api_key="temporary",
        model_ids=["gpt-5.4-nano"],
        client_factory=lambda _: client,
    )
    assert visible == ["gpt-5.4-nano", "unrelated-model"]
    assert client.closed

    with pytest.raises(RuntimeError, match="cannot access configured models"):
        validate_temporary_model_access(
            api_key="temporary",
            model_ids=["gpt-5.4-mini"],
            client_factory=lambda _: client,
        )


def test_preflight_failures_clear_keys_from_repository_tracebacks() -> None:
    temporary = "temporary-validation-marker"
    admin_project = "admin-project-validation-marker"
    admin_cost = "admin-cost-validation-marker"

    def fail_client(_: str) -> object:
        raise RuntimeError("construction failed")

    with pytest.raises(RuntimeError) as temporary_error:
        validate_temporary_model_access(
            api_key=temporary,
            model_ids=["gpt-5.4-nano"],
            client_factory=fail_client,
        )
    with pytest.raises(RuntimeError) as project_error:
        validate_admin_project_access(
            admin_api_key=admin_project,
            project_id="project",
            client_factory=fail_client,
        )
    with pytest.raises(RuntimeError) as cost_error:
        validate_filtered_cost_access(
            admin_api_key=admin_cost,
            execution_api_key_id="key",
            cost_window_start=1,
            client_factory=fail_client,
        )

    cases = (
        (temporary_error.value, temporary, "api_key", ""),
        (project_error.value, admin_project, "admin_api_key", ""),
        (cost_error.value, admin_cost, "admin_api_key", ""),
    )
    for error, marker, local_name, cleared_value in cases:
        frames = _repository_traceback_locals(error)
        assert marker not in repr(frames)
        assert all(frame.get(local_name) == cleared_value for frame in frames)
