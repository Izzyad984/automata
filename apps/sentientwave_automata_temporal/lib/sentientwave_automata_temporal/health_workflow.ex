defmodule SentientwaveAutomataTemporal.HealthWorkflow do
  @moduledoc false

  use TemporalSdk.Workflow

  @impl true
  def execute(_context, [_input]) do
    %{status: :ok}
  end
end
