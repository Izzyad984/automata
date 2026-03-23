defmodule SentientwaveAutomata.Agents.LLM.DeepResearchTest do
  use ExUnit.Case, async: true

  alias SentientwaveAutomata.Agents.LLM.Client

  test "stays standard when brave search is unavailable" do
    decision =
      Client.deep_research_decision(
        user_input: "Please do deep research on edge AI chips",
        available_tools: [%{name: "run_shell"}]
      )

    assert decision["enabled"] == false
    assert decision["mode"] == "standard"
  end

  test "turns on from explicit deep research flags with injected brave search availability" do
    decision =
      Client.deep_research_decision(
        user_input: "Help me draft a note",
        deep_research: true,
        available_tools: [%{name: "brave_search"}]
      )

    assert decision["enabled"] == true
    assert decision["mode"] == "deep_research"
    assert decision["requested_by_user"] == true

    assert decision["reason"] in [
             "explicit_user_request",
             "standard_response",
             "model_planner",
             "explicit_request"
           ]

    assert decision["queries"] != []
  end
end
