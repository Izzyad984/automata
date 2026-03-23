defmodule SentientwaveAutomataTemporal.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      SentientwaveAutomataTemporal.Bootstrap
    ]

    Supervisor.start_link(children,
      strategy: :one_for_one,
      name: SentientwaveAutomataTemporal.Supervisor
    )
  end
end
