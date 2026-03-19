defmodule SentientwaveAutomata.Agents.Skill do
  use Ecto.Schema
  import Ecto.Changeset

  alias SentientwaveAutomata.Agents.Skills.Parser

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "skills" do
    field :slug, :string
    field :name, :string
    field :markdown_body, :string
    field :enabled, :boolean, default: true
    field :metadata, :map, default: %{}

    has_many :designations, SentientwaveAutomata.Agents.SkillDesignation, foreign_key: :skill_id

    many_to_many :agents, SentientwaveAutomata.Agents.AgentProfile,
      join_through: SentientwaveAutomata.Agents.SkillDesignation,
      join_keys: [skill_id: :id, agent_id: :id]

    timestamps(type: :utc_datetime_usec)
  end

  def changeset(skill, attrs) do
    skill
    |> cast(attrs, [:slug, :name, :markdown_body, :enabled, :metadata])
    |> normalize_name_and_slug()
    |> validate_required([:slug, :name, :markdown_body])
    |> validate_length(:slug, min: 1, max: 160)
    |> validate_length(:name, min: 1, max: 200)
    |> validate_markdown()
    |> unique_constraint(:slug)
  end

  defp normalize_name_and_slug(changeset) do
    markdown = get_field(changeset, :markdown_body)
    parsed_name = parsed_name(markdown)

    name =
      case get_field(changeset, :name) do
        nil -> parsed_name || "Skill"
        "" -> parsed_name || "Skill"
        value -> String.trim(to_string(value))
      end

    slug =
      case get_field(changeset, :slug) do
        nil -> slugify(name)
        "" -> slugify(name)
        value -> slugify(value)
      end

    changeset
    |> put_change(:name, name)
    |> put_change(:slug, slug)
  end

  defp validate_markdown(changeset) do
    markdown = get_field(changeset, :markdown_body) || ""

    case Parser.parse(markdown) do
      {:ok, parsed} ->
        metadata =
          changeset
          |> get_field(:metadata)
          |> normalize_metadata()
          |> Map.put("parsed_name", parsed.name)
          |> Map.put("tools", parsed.tools)

        put_change(changeset, :metadata, metadata)

      {:error, :invalid_skill_markdown} ->
        add_error(changeset, :markdown_body, "must include a '# Skill:' header")

      {:error, reason} ->
        add_error(changeset, :markdown_body, "is invalid: #{inspect(reason)}")
    end
  end

  defp parsed_name(markdown) when is_binary(markdown) do
    case Parser.parse(markdown) do
      {:ok, parsed} -> parsed.name
      _ -> nil
    end
  end

  defp parsed_name(_), do: nil

  defp slugify(value) do
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

  defp normalize_metadata(value) when is_map(value), do: value
  defp normalize_metadata(_), do: %{}
end
