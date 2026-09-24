"""Check historical channel reconstruction against actual experiment send waves."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from mha_exp_level1.exp1_3 import runner


class WaveState(dict):
    """Provide the state attributes needed to execute a send wave without Docker."""

    def __getattr__(self, name):
        """Read test state fields using the module's attribute syntax."""
        return self[name]


class ChannelValidationTests(unittest.TestCase):
    """Exercise all real step methods and corrupt their recorded deliveries."""

    def setUp(self):
        """Generate one wave per module and retain the actual outbox method used."""
        classes = {
            runner.ACTUATOR: runner.TestActuator,
            runner.PERCEPTOR: runner.TestPerceptor,
            runner.LLREASONER: runner.TestLLReasoner,
            runner.KNOWLEDGE: runner.TestKnowledge,
            runner.HLREASONER: runner.TestHLReasoner,
            runner.GOALGRAPH: runner.TestGoalGraph,
            runner.MEMORY: runner.TestMemory,
            runner.LEARNER: runner.TestLearner,
        }
        directory = SimpleNamespace(internal=SimpleNamespace(**{
            attribute: [SimpleNamespace(module_id=f'{module}_{i}') for i in range(5)]
            for attribute, module in {
                'actuation': runner.ACTUATOR,
                'perception': runner.PERCEPTOR,
                'll_reasoning': runner.LLREASONER,
                'knowledge': runner.KNOWLEDGE,
                'hl_reasoning': runner.HLREASONER,
                'goals': runner.GOALGRAPH,
                'memory': runner.MEMORY,
                'learning': runner.LEARNER,
            }.items()
        }))
        self.states = {}
        self.actual_channels = {}
        for module, cls in classes.items():
            for index in range(5):
                sender = f'{module}_{index}'
                state = WaveState(time=0, directory=directory, outbox=Mock(), sent=[], received=[])
                cls(module_id=sender).step(state)
                self.states[sender] = {'sent': state['sent'], 'received': []}
                self.assertEqual(len(state['sent']), len(state.outbox.mock_calls))
                for message, call in zip(state['sent'], state.outbox.mock_calls):
                    recipient, payload, sent_ns = message
                    self.actual_channels[sender, recipient, payload, sent_ns] = (sender, recipient, call[0])
        for sender, state in self.states.items():
            for recipient, payload, sent_ns in state['sent']:
                self.states[recipient]['received'].append([sender, payload, sent_ns, sent_ns + 100])

    def test_reconstruction_matches_actual_outbox_calls(self):
        """Every inferred route must match the independently captured send method."""
        channels, waves = runner._reconstruct_channels(self.states)
        self.assertEqual(channels, self.actual_channels)
        self.assertEqual(len(set(channels.values())), 600)
        self.assertEqual(set(waves.values()), {1})
        self.assertTrue(runner.check_results(self.states))

    def test_rejects_missing_route_even_when_remaining_messages_match(self):
        """Removing one of two same-pair routes must fail despite exact delivery."""
        sender = 'llreasoner_0'
        removed = self.states[sender]['sent'][30:35]
        del self.states[sender]['sent'][30:35]
        for recipient, payload, sent_ns in removed:
            self.states[recipient]['received'].remove([sender, payload, sent_ns, sent_ns + 100])
        self.assertFalse(runner.check_results(self.states))

    def test_rejects_wrong_recipient_with_matching_receipt(self):
        """An omitted concrete pair cannot hide behind another receiver of its type."""
        sender = 'actuator_0'
        _, payload, sent_ns = self.states[sender]['sent'][0]
        self.states[sender]['sent'][0] = ('llreasoner_1', payload, sent_ns)
        receipt = [sender, payload, sent_ns, sent_ns + 100]
        self.states['llreasoner_0']['received'].remove(receipt)
        self.states['llreasoner_1']['received'].append(receipt)
        self.assertFalse(runner.check_results(self.states))

    def test_rejects_missing_and_duplicate_receipts(self):
        """Reconstruction must retain exact-once matching of recorded messages."""
        receipts = self.states['actuator_0']['received']
        removed = receipts.pop()
        self.assertFalse(runner.check_results(self.states))
        receipts.extend([removed, removed])
        self.assertFalse(runner.check_results(self.states))

    def test_rejects_ambiguous_message_identity(self):
        """Two same-pair routes cannot share an indistinguishable saved identity."""
        sends = self.states['llreasoner_0']['sent']
        recipient, payload, sent_ns = sends[15]
        sends[20] = (recipient, payload, sent_ns)
        with self.assertRaises(ValueError):
            runner._reconstruct_channels(self.states)


if __name__ == '__main__':
    unittest.main()
