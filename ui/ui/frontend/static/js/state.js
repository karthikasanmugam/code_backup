export const state = {
  policies: {},
  datasets: [],
  checkpoints: [],
  selectedPolicy: '',
  selectedDatasetId: '',
  selectedCheckpointPath: '',
  training: null,
  system: null,
  inferenceMetadata: null,
  serverRunning: false,
  bridgeRunning: false,
  viewer: {
    datasetId: '', metadata: null, episodes: [], total: 0, offset: 0,
    task: '', episode: null, actions: null, videos: [], actionPage: 0,
    requestToken: 0,
  },
  evaluation: { jobId: '', loaded: null, pollTimer: null, result: null },
};
