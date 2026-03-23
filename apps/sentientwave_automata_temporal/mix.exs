defmodule SentientwaveAutomataTemporal.MixProject do
  use Mix.Project

  def project do
    [
      app: :sentientwave_automata_temporal,
      version: "0.1.0",
      build_path: "../../_build",
      config_path: "../../config/config.exs",
      deps_path: "../../deps",
      lockfile: "../../mix.lock",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {SentientwaveAutomataTemporal.Application, []}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:sentientwave_automata, in_umbrella: true},
      {:temporal_sdk, "~> 0.1.17"}
    ]
  end
end
