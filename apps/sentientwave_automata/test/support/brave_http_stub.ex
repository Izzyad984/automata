defmodule SentientwaveAutomata.TestSupport.BraveHTTPStub do
  @moduledoc false

  def get_json(url, headers, opts \\ []) do
    send(test_pid(), {:brave_http_request, %{url: url, headers: headers, opts: opts}})

    query =
      url
      |> URI.parse()
      |> Map.get(:query, "")
      |> URI.decode_query()
      |> Map.get("q", "")

    case Map.fetch(responses(), query) do
      {:ok, response} -> response
      :error -> {:error, {:missing_fixture, query}}
    end
  end

  defp responses, do: Process.get(:brave_http_responses, %{})

  defp test_pid do
    Process.get(:brave_http_test_pid, self())
  end
end
