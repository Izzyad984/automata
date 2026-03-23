defmodule SentientwaveAutomata.Agents.DeepResearchTest do
  use ExUnit.Case, async: true

  alias SentientwaveAutomata.Agents.DeepResearch

  test "treats explicit deep research requests as eligible when brave search is available" do
    assert DeepResearch.should_consider?(
             "Please do deep research on the latest robotics startups in Japan",
             [%{name: "brave_search"}]
           )

    refute DeepResearch.should_consider?(
             "Please do deep research on the latest robotics startups in Japan",
             [%{name: "run_shell"}]
           )
  end

  test "identifies complex current-events style prompts as deep research candidates" do
    prompt =
      "Compare the latest open source agent frameworks, analyze their tradeoffs, and give me a current landscape report with sources."

    assert DeepResearch.complexity_candidate?(prompt)
  end

  test "normalizes deep research decisions with bounded rounds and queries" do
    decision =
      DeepResearch.normalize_decision(
        %{
          "enabled" => "true",
          "requested_by_user" => "yes",
          "reason" => "model_planner",
          "max_rounds" => 99,
          "queries" => [
            "alpha",
            "beta",
            "gamma",
            "delta"
          ]
        },
        "do deep research on alpha",
        [%{name: "brave_search"}]
      )

    assert decision["enabled"] == true
    assert decision["requested_by_user"] == true
    assert decision["reason"] == "model_planner"
    assert decision["max_rounds"] <= DeepResearch.config()["max_rounds"]
    assert length(decision["queries"]) <= DeepResearch.config()["max_queries_per_round"]
  end

  test "normalizes round reviews and falls back to evidence sources" do
    evidence = [
      %{
        "query" => "alpha",
        "findings" => ["Alpha is growing quickly", "Two notable launches happened this quarter"],
        "sources" => [
          %{
            "title" => "Alpha News",
            "url" => "https://example.com/a",
            "summary" => "Launch summary"
          },
          %{
            "title" => "Alpha Blog",
            "url" => "https://example.com/b",
            "summary" => "Market analysis"
          }
        ]
      }
    ]

    review =
      DeepResearch.normalize_round_review(
        %{
          "continue_research" => true,
          "follow_up_queries" => ["alpha valuation", "alpha customers"],
          "round_summary" => "The first pass found strong launch momentum."
        },
        evidence,
        1,
        2
      )

    assert review["continue_research"] == true
    assert review["round_summary"] == "The first pass found strong launch momentum."
    assert review["follow_up_queries"] == ["alpha valuation", "alpha customers"]

    assert Enum.map(review["top_sources"], & &1["url"]) == [
             "https://example.com/a",
             "https://example.com/b"
           ]
  end
end
