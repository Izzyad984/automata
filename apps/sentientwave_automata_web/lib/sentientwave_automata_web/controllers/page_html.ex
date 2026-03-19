defmodule SentientwaveAutomataWeb.PageHTML do
  @moduledoc """
  This module contains pages rendered by PageController.

  See the `page_html` directory for all templates available.
  """
  use SentientwaveAutomataWeb, :html

  alias SentientwaveAutomataWeb.Layouts

  attr :flash, :map, required: true
  attr :title, :string, required: true
  attr :subtitle, :string, default: ""
  attr :status, :map, required: true
  attr :admin_user, :string, required: true
  attr :nav, :list, required: true
  slot :inner_block, required: true

  def admin_shell(assigns) do
    ~H"""
    <Layouts.flash_group flash={@flash} />

    <div class="sw-admin-shell">
      <aside class="sw-sidebar">
        <div class="sw-brand">
          <p class="sw-brand-kicker">SentientWave Automata</p>
          <h2 class="sw-brand-title">{@status.company_name}</h2>
          <p class="sw-brand-subtitle">{@status.group_name}</p>
        </div>

        <nav class="sw-nav" aria-label="Admin navigation">
          <%= for item <- @nav do %>
            <a href={item.href} class={["sw-nav-link", item.active && "is-active"]}>
              {item.label}
            </a>
          <% end %>
        </nav>

        <div class="sw-sidebar-meta">
          <p>Admin: <strong>{@admin_user}</strong></p>
          <p>Source: <strong>{@status.source}</strong></p>
          <p>Homeserver: <strong>{@status.homeserver_domain}</strong></p>
        </div>

        <div class="sw-sidebar-theme">
          <p class="sw-sidebar-section-title">Appearance</p>
          <Layouts.theme_toggle />
        </div>

        <form action={~p"/logout"} method="post" class="sw-sidebar-logout">
          <input type="hidden" name="_csrf_token" value={Plug.CSRFProtection.get_csrf_token()} />
          <input type="hidden" name="_method" value="delete" />
          <button type="submit" class="sw-btn sw-btn-ghost sw-btn-block">Sign Out</button>
        </form>
      </aside>

      <main class="sw-main">
        <header class="sw-page-header">
          <div>
            <p class="sw-page-kicker">Internal Admin Console</p>
            <h1 class="sw-page-title">{@title}</h1>
            <p :if={@subtitle != ""} class="sw-page-subtitle">{@subtitle}</p>
          </div>

          <div class="sw-status-row">
            <span class={["sw-pill", service_class(@status.services.automata)]}>
              Automata: {@status.services.automata}
            </span>
            <span class={["sw-pill", service_class(@status.services.matrix)]}>
              Matrix: {@status.services.matrix}
            </span>
            <span class={["sw-pill", service_class(@status.services.temporal_ui)]}>
              Temporal: {@status.services.temporal_ui}
            </span>
          </div>
        </header>

        <section class="sw-main-content">
          {render_slot(@inner_block)}
        </section>
      </main>
    </div>
    """
  end

  def trace_status_class("ok"), do: "is-ok"
  def trace_status_class("error"), do: "is-issue"
  def trace_status_class(_), do: "is-neutral"

  def format_timestamp(%DateTime{} = timestamp) do
    Calendar.strftime(timestamp, "%Y-%m-%d %H:%M:%S UTC")
  end

  def format_timestamp(_), do: "not recorded"

  def trace_requester_name(trace) do
    cond do
      present?(trace.requester_display_name) -> trace.requester_display_name
      present?(trace.requester_localpart) -> trace.requester_localpart
      present?(trace.requester_mxid) -> trace.requester_mxid
      true -> "Unknown requester"
    end
  end

  def trace_requester_meta(trace) do
    [trace.requester_kind, trace.requester_mxid]
    |> Enum.filter(&present?/1)
    |> Enum.join(" · ")
  end

  def trace_preview(trace) do
    request_preview = request_message_preview(trace)
    response_preview = get_in(trace.response_payload || %{}, ["content"])
    error_preview = get_in(trace.error_payload || %{}, ["reason"])

    cond do
      present?(request_preview) -> truncate_text(request_preview, 140)
      present?(response_preview) -> truncate_text(response_preview, 140)
      present?(error_preview) -> truncate_text(error_preview, 140)
      true -> "No preview available."
    end
  end

  def trace_duration(trace) do
    case {trace.requested_at, trace.completed_at} do
      {%DateTime{} = requested_at, %DateTime{} = completed_at} ->
        diff = DateTime.diff(completed_at, requested_at, :millisecond)
        "#{max(diff, 0)} ms"

      _ ->
        "n/a"
    end
  end

  def pretty_json(nil), do: "No payload recorded."

  def pretty_json(payload) do
    Jason.encode!(payload, pretty: true)
  rescue
    _ -> inspect(payload, pretty: true, limit: :infinity)
  end

  def active_designation_count(skill) do
    skill.designations
    |> Enum.count(&(&1.status == :active))
  end

  def designation_status_class(:active), do: "is-ok"
  def designation_status_class(:rolled_back), do: "is-neutral"
  def designation_status_class(_), do: "is-neutral"

  def skill_summary(skill) do
    skill.metadata
    |> Map.get("summary", "")
    |> case do
      value when is_binary(value) and value != "" -> value
      _ -> "No summary added yet."
    end
  end

  def skill_tags(skill) do
    skill.metadata
    |> Map.get("tags", [])
    |> case do
      tags when is_list(tags) -> tags
      _ -> []
    end
  end

  def skill_tools(skill) do
    skill.metadata
    |> Map.get("tools", [])
    |> case do
      tools when is_list(tools) -> tools
      _ -> []
    end
  end

  def designation_agent_name(designation) do
    agent = designation.agent
    display_name = agent && agent.display_name
    slug = agent && agent.slug

    cond do
      present?(display_name) -> display_name
      present?(slug) -> slug
      true -> "Unknown agent"
    end
  end

  defp service_class(status) when is_binary(status) do
    cond do
      String.starts_with?(status, "ok") -> "is-ok"
      status == "skipped" -> "is-neutral"
      true -> "is-issue"
    end
  end

  defp request_message_preview(trace) do
    trace.request_payload
    |> case do
      %{"messages" => messages} when is_list(messages) ->
        messages
        |> Enum.reverse()
        |> Enum.find_value(fn
          %{"role" => "user", "content" => content} when is_binary(content) -> content
          _ -> nil
        end)

      _ ->
        nil
    end
  end

  defp truncate_text(text, max_length) when is_binary(text) do
    trimmed = String.trim(text)

    if String.length(trimmed) > max_length do
      String.slice(trimmed, 0, max_length - 1) <> "…"
    else
      trimmed
    end
  end

  defp present?(value) when is_binary(value), do: String.trim(value) != ""
  defp present?(nil), do: false
  defp present?(_), do: true

  embed_templates "page_html/*"
end
