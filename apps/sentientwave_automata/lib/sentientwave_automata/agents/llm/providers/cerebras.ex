defmodule SentientwaveAutomata.Agents.LLM.Providers.Cerebras do
  @moduledoc false

  @behaviour SentientwaveAutomata.Agents.LLMProvider

  alias SentientwaveAutomata.Agents.LLM.HTTP

  @default_model "gpt-oss-120b"
  @default_base_url "https://api.cerebras.ai/v1"
  @default_version_patch "2"

  @impl true
  def complete(messages, opts \\ []) when is_list(messages) do
    model = Keyword.get(opts, :model, System.get_env("AUTOMATA_LLM_MODEL", @default_model))

    base_url =
      Keyword.get(opts, :base_url, System.get_env("AUTOMATA_LLM_API_BASE", @default_base_url))

    api_key =
      Keyword.get(
        opts,
        :api_key,
        System.get_env("AUTOMATA_LLM_API_KEY", System.get_env("CEREBRAS_API_KEY", ""))
      )

    if String.trim(api_key) == "" do
      {:error, :missing_api_key}
    else
      url = String.trim_trailing(base_url, "/") <> "/chat/completions"

      payload = %{
        "model" => model,
        "messages" => messages,
        "temperature" => 0.2
      }

      headers =
        [{"authorization", "Bearer " <> api_key}]
        |> maybe_add_version_patch_header(version_patch(opts))

      with {:ok, status, body} <- HTTP.post_json(url, headers, payload, opts),
           {:ok, text} <- handle_response(status, body) do
        {:ok, text}
      else
        {:error, reason} -> {:error, reason}
      end
    end
  end

  defp handle_response(status, body) when status in 200..299, do: extract_text(body)
  defp handle_response(status, body), do: {:error, {:http_error, status, body}}

  defp extract_text(%{"choices" => [%{"message" => %{"content" => content}} | _]})
       when is_binary(content) do
    {:ok, String.trim(content)}
  end

  defp extract_text(body), do: {:error, {:invalid_response, body}}

  defp version_patch(opts) do
    Keyword.get(
      opts,
      :version_patch,
      System.get_env("AUTOMATA_LLM_CEREBRAS_VERSION_PATCH", @default_version_patch)
    )
  end

  defp maybe_add_version_patch_header(headers, value) when is_binary(value) do
    case String.trim(value) do
      "" -> headers
      version_patch -> headers ++ [{"x-cerebras-version-patch", version_patch}]
    end
  end

  defp maybe_add_version_patch_header(headers, _), do: headers
end
