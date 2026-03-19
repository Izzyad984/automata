defmodule SentientwaveAutomata.Agents.SkillCatalogTest do
  use SentientwaveAutomata.DataCase, async: false

  alias SentientwaveAutomata.Agents

  test "creates and updates a global skill with parsed metadata" do
    assert {:ok, skill} =
             Agents.create_skill(%{
               "name" => "Inbox Triage",
               "slug" => "inbox-triage",
               "markdown_body" =>
                 skill_markdown("Inbox Triage", ["summarize requests", "propose next steps"]),
               "enabled" => true,
               "metadata" => %{"summary" => "Organize new requests", "tags" => ["ops", "triage"]}
             })

    assert skill.slug == "inbox-triage"
    assert skill.enabled
    assert skill.metadata["parsed_name"] == "Inbox Triage"
    assert skill.metadata["tools"] == ["summarize requests", "propose next steps"]
    assert skill.metadata["summary"] == "Organize new requests"

    assert {:ok, updated} =
             Agents.update_skill(skill, %{
               "name" => "Inbox Triage Updated",
               "slug" => "inbox-triage-updated",
               "markdown_body" =>
                 skill_markdown("Inbox Triage Updated", ["capture missing info", "route work"]),
               "enabled" => false,
               "metadata" => %{"summary" => "Updated summary", "tags" => ["ops"]}
             })

    assert updated.slug == "inbox-triage-updated"
    refute updated.enabled
    assert updated.metadata["parsed_name"] == "Inbox Triage Updated"
    assert updated.metadata["tools"] == ["capture missing info", "route work"]
    assert updated.metadata["summary"] == "Updated summary"
  end

  test "designates and rolls back skills per agent while preserving history" do
    assert {:ok, agent} =
             Agents.upsert_agent(%{
               slug: "catalog-agent",
               kind: :agent,
               display_name: "Catalog Agent",
               matrix_localpart: "catalog-agent",
               status: :active
             })

    assert {:ok, skill} =
             Agents.create_skill(%{
               "name" => "Delivery Planner",
               "markdown_body" =>
                 skill_markdown("Delivery Planner", ["plan delivery", "identify blockers"]),
               "enabled" => true
             })

    assert [] == Agents.list_agent_skills(agent.id)

    assert {:ok, designation} =
             Agents.designate_skill(skill.id, agent.id, %{metadata: %{"source" => "test"}})

    [effective_skill] = Agents.list_agent_skills(agent.id)
    assert effective_skill.id == skill.id

    [history_entry] = Agents.list_skill_designations(skill.id)
    assert history_entry.id == designation.id
    assert history_entry.status == :active
    assert history_entry.agent_id == agent.id

    assert {:ok, rolled_back} = Agents.rollback_skill_designation(designation.id, %{})
    assert rolled_back.status == :rolled_back
    assert rolled_back.rolled_back_at

    assert [] == Agents.list_agent_skills(agent.id)
    assert 1 == Agents.count_skill_designations(skill.id)
    assert 0 == Agents.count_skill_designations(skill.id, status: :active)
  end

  test "filters out disabled skills from effective agent skills" do
    assert {:ok, agent} =
             Agents.upsert_agent(%{
               slug: "disabled-skill-agent",
               kind: :agent,
               display_name: "Disabled Skill Agent",
               matrix_localpart: "disabled-skill-agent",
               status: :active
             })

    assert {:ok, skill} =
             Agents.create_skill(%{
               "name" => "Drafting",
               "markdown_body" => skill_markdown("Drafting", ["draft response"]),
               "enabled" => true
             })

    assert {:ok, _designation} = Agents.designate_skill(skill.id, agent.id, %{})
    assert [_] = Agents.list_agent_skills(agent.id)

    assert {:ok, _updated} = Agents.update_skill(skill, %{"enabled" => false})
    assert [] == Agents.list_agent_skills(agent.id)
  end

  defp skill_markdown(name, bullets) do
    bullet_lines =
      bullets
      |> Enum.map_join("\n", fn bullet -> "- #{bullet}" end)

    """
    # Skill: #{name}

    This skill helps the agent stay consistent.

    #{bullet_lines}
    """
  end
end
