import Config

# config/runtime.exs is executed for all environments, including
# during releases. It is executed after compilation and before the
# system starts, so it is typically used to load production configuration
# and secrets from environment variables or elsewhere. Do not define
# any compile-time configuration in here, as it won't be applied.
# The block below contains prod specific runtime configuration.

config :sentientwave_automata_web, SentientwaveAutomataWeb.Endpoint,
  http: [port: String.to_integer(System.get_env("PORT", "4000"))]

matrix_adapter =
  case System.get_env("MATRIX_ADAPTER", "local") do
    "synapse" -> SentientwaveAutomata.Adapters.Matrix.Synapse
    _ -> SentientwaveAutomata.Adapters.Matrix.Local
  end

parse_temporal_endpoint = fn address ->
  case String.split(address, ":", parts: 2) do
    [host, port] ->
      {String.to_charlist(host), String.to_integer(port)}

    [host] ->
      {String.to_charlist(host), 7233}
  end
end

temporal_address = System.get_env("TEMPORAL_ADDRESS", "127.0.0.1:7233")
{temporal_host, temporal_port} = parse_temporal_endpoint.(temporal_address)
temporal_namespace = System.get_env("TEMPORAL_NAMESPACE", "default")

temporal_workflow_task_queue =
  System.get_env("AUTOMATA_TEMPORAL_WORKFLOW_TASK_QUEUE", "automata-workflows")

temporal_activity_task_queue =
  System.get_env("AUTOMATA_TEMPORAL_ACTIVITY_TASK_QUEUE", "automata-activities")

temporal_worker_identity_prefix =
  System.get_env("AUTOMATA_TEMPORAL_WORKER_IDENTITY_PREFIX", "automata")

config :temporal_sdk,
  node: %{scope_config: [automata: 10]},
  clusters: [
    automata: [
      client: %{
        adapter:
          {:temporal_sdk_grpc_adapter_gun_pool,
           [endpoints: [{temporal_host, temporal_port}], pool_size: 5]},
        grpc_opts: [timeout: 2_000],
        grpc_opts_longpoll: [timeout: 70_000]
      },
      workflows: [[task_queue: temporal_workflow_task_queue]],
      activities: [[task_queue: temporal_activity_task_queue]]
    ]
  ]

config :sentientwave_automata,
  temporal_cluster: :automata,
  temporal_namespace: temporal_namespace,
  temporal_workflow_task_queue: temporal_workflow_task_queue,
  temporal_activity_task_queue: temporal_activity_task_queue,
  temporal_worker_identity_prefix: temporal_worker_identity_prefix,
  deep_research_max_rounds:
    String.to_integer(System.get_env("AUTOMATA_DEEP_RESEARCH_MAX_ROUNDS", "2")),
  deep_research_max_queries_per_round:
    String.to_integer(System.get_env("AUTOMATA_DEEP_RESEARCH_MAX_QUERIES_PER_ROUND", "3")),
  deep_research_results_per_query:
    String.to_integer(System.get_env("AUTOMATA_DEEP_RESEARCH_RESULTS_PER_QUERY", "5")),
  embedding_dim: String.to_integer(System.get_env("AUTOMATA_EMBEDDING_DIM", "64")),
  agent_skills_path: System.get_env("AUTOMATA_SKILLS_PATH", "skills"),
  llm_provider: System.get_env("AUTOMATA_LLM_PROVIDER", "local"),
  llm_model: System.get_env("AUTOMATA_LLM_MODEL", "local-default"),
  llm_api_base: System.get_env("AUTOMATA_LLM_API_BASE", ""),
  matrix_adapter: matrix_adapter

if config_env() == :prod do
  database_url =
    System.get_env("DATABASE_URL") ||
      raise """
      environment variable DATABASE_URL is missing.
      For example: ecto://USER:PASS@HOST/DATABASE
      """

  maybe_ipv6 = if System.get_env("ECTO_IPV6") in ~w(true 1), do: [:inet6], else: []

  config :sentientwave_automata, SentientwaveAutomata.Repo,
    # ssl: true,
    url: database_url,
    pool_size: String.to_integer(System.get_env("POOL_SIZE") || "10"),
    # For machines with several cores, consider starting multiple pools of `pool_size`
    # pool_count: 4,
    socket_options: maybe_ipv6

  # The secret key base is used to sign/encrypt cookies and other secrets.
  # A default value is used in config/dev.exs and config/test.exs but you
  # want to use a different value for prod and you most likely don't want
  # to check this value into version control, so we use an environment
  # variable instead.
  secret_key_base =
    System.get_env("SECRET_KEY_BASE") ||
      raise """
      environment variable SECRET_KEY_BASE is missing.
      You can generate one by calling: mix phx.gen.secret
      """

  config :sentientwave_automata_web, SentientwaveAutomataWeb.Endpoint,
    http: [
      # Enable IPv6 and bind on all interfaces.
      # Set it to  {0, 0, 0, 0, 0, 0, 0, 1} for local network only access.
      ip: {0, 0, 0, 0, 0, 0, 0, 0}
    ],
    secret_key_base: secret_key_base

  # ## Using releases
  #
  # If you are doing OTP releases, you need to instruct Phoenix
  # to start each relevant endpoint:
  #
  #     config :sentientwave_automata_web, SentientwaveAutomataWeb.Endpoint, server: true
  #
  # Then you can assemble a release by calling `mix release`.
  # See `mix help release` for more information.

  # ## SSL Support
  #
  # To get SSL working, you will need to add the `https` key
  # to your endpoint configuration:
  #
  #     config :sentientwave_automata_web, SentientwaveAutomataWeb.Endpoint,
  #       https: [
  #         ...,
  #         port: 443,
  #         cipher_suite: :strong,
  #         keyfile: System.get_env("SOME_APP_SSL_KEY_PATH"),
  #         certfile: System.get_env("SOME_APP_SSL_CERT_PATH")
  #       ]
  #
  # The `cipher_suite` is set to `:strong` to support only the
  # latest and more secure SSL ciphers. This means old browsers
  # and clients may not be supported. You can set it to
  # `:compatible` for wider support.
  #
  # `:keyfile` and `:certfile` expect an absolute path to the key
  # and cert in disk or a relative path inside priv, for example
  # "priv/ssl/server.key". For all supported SSL configuration
  # options, see https://hexdocs.pm/plug/Plug.SSL.html#configure/1
  #
  # We also recommend setting `force_ssl` in your config/prod.exs,
  # ensuring no data is ever sent via http, always redirecting to https:
  #
  #     config :sentientwave_automata_web, SentientwaveAutomataWeb.Endpoint,
  #       force_ssl: [hsts: true]
  #
  # Check `Plug.SSL` for all available options in `force_ssl`.

  config :sentientwave_automata, :dns_cluster_query, System.get_env("DNS_CLUSTER_QUERY")
end
