const backendEnabled = bool.fromEnvironment(
  'BACKEND_ENABLED',
  defaultValue: false,
);

const configuredServerBaseUrl = String.fromEnvironment('SERVER_BASE_URL');
