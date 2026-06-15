from unittest.mock import MagicMock


class TestDiagnosisAgent:
    def test_diagnosis_prompt_enforces_ordered_todos_with_review(self):
        from src.agents.diagnosis import DiagnosisAgent

        backend = MagicMock()
        response = MagicMock()
        response.text = (
            "TODO 1\n"
            "LOCATION: line 3\n"
            "SNIPPET: `if (rst)`\n"
            "BUG: bad reset\n"
            "FIX: invert reset\n"
            "REVIEW: verify reset polarity\n"
        )
        backend.generate.return_value = response

        agent = DiagnosisAgent(backend)
        result = agent.execute(
            rtl_files={"rtl.sv": "module top; endmodule"},
            config={"errors": [{"line": 3, "message": "bad reset"}], "spec": "counter"},
        )

        assert "TODO 1" in result.raw_output
        call_kwargs = backend.generate.call_args.kwargs
        assert "ordered TODO list" in call_kwargs["system_prompt"]
        assert "LOCATION:" in call_kwargs["system_prompt"]
        assert "BUG:" in call_kwargs["system_prompt"]
        assert "FIX:" in call_kwargs["system_prompt"]
        assert "tools" not in call_kwargs
