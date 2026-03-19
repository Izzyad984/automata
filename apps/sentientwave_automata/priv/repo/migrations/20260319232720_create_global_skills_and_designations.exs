defmodule SentientwaveAutomata.Repo.Migrations.CreateGlobalSkillsAndDesignations do
  use Ecto.Migration

  def up do
    create table(:skills, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :slug, :string, null: false
      add :name, :string, null: false
      add :markdown_body, :text, null: false
      add :enabled, :boolean, null: false, default: true
      add :metadata, :map, null: false, default: %{}

      timestamps(type: :utc_datetime_usec)
    end

    create unique_index(:skills, [:slug])

    create table(:agent_skill_designations, primary_key: false) do
      add :id, :binary_id, primary_key: true

      add :skill_id, references(:skills, type: :binary_id, on_delete: :delete_all), null: false
      add :agent_id, references(:agents, type: :binary_id, on_delete: :delete_all), null: false
      add :status, :string, null: false, default: "active"
      add :designated_at, :utc_datetime_usec, null: false
      add :rolled_back_at, :utc_datetime_usec
      add :metadata, :map, null: false, default: %{}

      timestamps(type: :utc_datetime_usec)
    end

    create index(:agent_skill_designations, [:skill_id, :status])
    create index(:agent_skill_designations, [:agent_id, :status])

    create unique_index(
             :agent_skill_designations,
             [:skill_id, :agent_id],
             where: "status = 'active'",
             name: :agent_skill_designations_active_skill_agent_index
           )

    flush()
    backfill_legacy_agent_skills()
  end

  def down do
    drop table(:agent_skill_designations)
    drop table(:skills)
  end

  defp backfill_legacy_agent_skills do
    rows =
      repo().query!("""
      SELECT
        id,
        agent_id,
        name,
        markdown_body,
        enabled,
        metadata,
        inserted_at,
        updated_at
      FROM agent_skills
      ORDER BY inserted_at ASC, id ASC
      """).rows

    {skills, designations, _slug_set, _skill_map} =
      Enum.reduce(rows, {[], [], MapSet.new(), %{}}, fn [
                                                          legacy_id,
                                                          agent_id,
                                                          name,
                                                          markdown_body,
                                                          enabled,
                                                          metadata,
                                                          inserted_at,
                                                          updated_at
                                                        ],
                                                        {skill_rows, designation_rows, slug_set,
                                                         skill_map} ->
        key = {name, markdown_body}

        {skill_id, skill_rows, slug_set, skill_map} =
          case Map.get(skill_map, key) do
            nil ->
              parsed_metadata = build_parsed_metadata(name, markdown_body, metadata)
              slug = unique_slug(name, slug_set)
              skill_id = Ecto.UUID.generate()

              skill_row = %{
                id: skill_id,
                slug: slug,
                name: name || slug,
                markdown_body: markdown_body || "",
                enabled: true,
                metadata: Map.put(parsed_metadata, "legacy_backfill", true),
                inserted_at: inserted_at || DateTime.utc_now(),
                updated_at: updated_at || inserted_at || DateTime.utc_now()
              }

              {
                skill_id,
                [skill_row | skill_rows],
                MapSet.put(slug_set, slug),
                Map.put(skill_map, key, skill_id)
              }

            existing_skill_id ->
              {existing_skill_id, skill_rows, slug_set, skill_map}
          end

        designation_time = inserted_at || DateTime.utc_now()

        designation_row = %{
          id: Ecto.UUID.generate(),
          skill_id: skill_id,
          agent_id: agent_id,
          status: if(enabled, do: "active", else: "rolled_back"),
          designated_at: designation_time,
          rolled_back_at: if(enabled, do: nil, else: updated_at || designation_time),
          metadata: %{"legacy_skill_id" => legacy_id, "legacy_backfill" => true},
          inserted_at: designation_time,
          updated_at: updated_at || designation_time
        }

        {skill_rows, [designation_row | designation_rows], slug_set, skill_map}
      end)

    if skills != [] do
      repo().insert_all("skills", skills)
    end

    if designations != [] do
      repo().insert_all("agent_skill_designations", designations)
    end
  end

  defp unique_slug(name, slug_set) do
    base_slug = normalize_slug(name)

    Stream.iterate(1, &(&1 + 1))
    |> Enum.find_value(fn
      1 ->
        if MapSet.member?(slug_set, base_slug), do: nil, else: base_slug

      index ->
        candidate = "#{base_slug}-#{index}"
        if MapSet.member?(slug_set, candidate), do: nil, else: candidate
    end)
  end

  defp normalize_slug(value) do
    value
    |> to_string()
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9]+/u, "-")
    |> String.replace(~r/^-+|-+$/u, "")
    |> case do
      "" -> "skill"
      slug -> slug
    end
  end

  defp build_parsed_metadata(name, markdown_body, metadata) do
    tools =
      markdown_body
      |> to_string()
      |> String.split("\n")
      |> Enum.filter(&String.starts_with?(String.trim(&1), "- "))
      |> Enum.map(&String.trim_leading(String.trim(&1), "- "))

    parsed_name =
      markdown_body
      |> to_string()
      |> String.split("\n")
      |> Enum.find_value(fn
        "# Skill:" <> skill_name -> String.trim(skill_name)
        _ -> nil
      end)

    metadata_map =
      case metadata do
        value when is_map(value) -> value
        _ -> %{}
      end

    metadata_map
    |> Map.put("parsed_name", parsed_name || name || "Skill")
    |> Map.put("tools", tools)
  end
end
