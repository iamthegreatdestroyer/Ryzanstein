<script lang="ts">
  import { onMount } from "svelte";
  import { ListModels, LoadModel, UnloadModel } from "../../wailsjs/go/main/App";
  import { EventsOn } from "../../wailsjs/runtime/runtime";

  export let config = null;

  interface Model {
    id: string;
    name: string;
    size: string;
    contextLength: number;
    loaded: boolean;
    status: string;
  }

  let models: Model[] = [];
  let isLoading = true;
  let loadingModelId: string | null = null;

  onMount(async () => {
    await refreshModels();

    EventsOn("model:loaded", (data: any) => {
      loadingModelId = null;
      refreshModels();
    });

    EventsOn("model:loadError", (data: any) => {
      loadingModelId = null;
      console.error("Model load error:", data.error);
      refreshModels();
    });
  });

  async function refreshModels() {
    isLoading = true;
    try {
      models = await ListModels();
    } catch (error) {
      console.error("Failed to list models:", error);
    } finally {
      isLoading = false;
    }
  }

  async function handleLoadModel(modelId: string) {
    loadingModelId = modelId;
    try {
      await LoadModel(modelId);
    } catch (error) {
      console.error("Failed to load model:", error);
      loadingModelId = null;
    }
  }

  async function handleUnloadModel(modelId: string) {
    try {
      await UnloadModel(modelId);
      await refreshModels();
    } catch (error) {
      console.error("Failed to unload model:", error);
    }
  }

  function formatSize(size: string): string {
    return size || "Unknown";
  }

  function formatContext(length: number): string {
    if (length >= 1000) {
      return `${(length / 1000).toFixed(0)}K`;
    }
    return `${length}`;
  }
</script>

<div class="model-panel">
  <div class="panel-header">
    <h2>Model Management</h2>
    <button class="refresh-btn" on:click={refreshModels} disabled={isLoading}>
      {isLoading ? "Loading..." : "Refresh"}
    </button>
  </div>

  {#if isLoading && models.length === 0}
    <div class="loading-state">
      <p>Loading models...</p>
    </div>
  {:else if models.length === 0}
    <div class="empty-state">
      <p>No models found.</p>
      <p class="hint">
        Download models with: <code>python scripts/download_models.py</code>
      </p>
    </div>
  {:else}
    <div class="model-grid">
      {#each models as model (model.id)}
        <div class="model-card" class:loaded={model.loaded}>
          <div class="model-header">
            <h3>{model.name}</h3>
            <span class="status-badge" class:active={model.loaded}>
              {model.loaded ? "Loaded" : "Available"}
            </span>
          </div>

          <div class="model-details">
            <div class="detail">
              <span class="label">ID</span>
              <span class="value">{model.id}</span>
            </div>
            <div class="detail">
              <span class="label">Size</span>
              <span class="value">{formatSize(model.size)}</span>
            </div>
            <div class="detail">
              <span class="label">Context</span>
              <span class="value">{formatContext(model.contextLength)}</span>
            </div>
            <div class="detail">
              <span class="label">Status</span>
              <span class="value">{model.status}</span>
            </div>
          </div>

          <div class="model-actions">
            {#if model.loaded}
              <button
                class="btn btn-unload"
                on:click={() => handleUnloadModel(model.id)}
              >
                Unload
              </button>
            {:else}
              <button
                class="btn btn-load"
                on:click={() => handleLoadModel(model.id)}
                disabled={loadingModelId === model.id}
              >
                {loadingModelId === model.id ? "Loading..." : "Load Model"}
              </button>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .model-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    gap: 16px;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .panel-header h2 {
    margin: 0;
    font-size: 18px;
  }

  .refresh-btn {
    padding: 6px 16px;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 4px;
    color: #00d4ff;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s;
  }

  .refresh-btn:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.25);
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .loading-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    color: #888;
  }

  .hint {
    font-size: 12px;
    color: #666;
    margin-top: 8px;
  }

  .hint code {
    background: rgba(0, 0, 0, 0.3);
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 11px;
  }

  .model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
    overflow-y: auto;
    padding: 4px;
  }

  .model-card {
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
    transition: all 0.2s;
  }

  .model-card:hover {
    border-color: rgba(255, 255, 255, 0.15);
    background: rgba(0, 0, 0, 0.3);
  }

  .model-card.loaded {
    border-color: rgba(0, 200, 100, 0.3);
  }

  .model-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .model-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
  }

  .status-badge {
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
    background: rgba(255, 255, 255, 0.08);
    color: #888;
  }

  .status-badge.active {
    background: rgba(0, 200, 100, 0.2);
    color: #00c864;
  }

  .model-details {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 16px;
  }

  .detail {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .label {
    font-size: 11px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .value {
    font-size: 13px;
    color: #ccc;
  }

  .model-actions {
    display: flex;
    gap: 8px;
  }

  .btn {
    flex: 1;
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s;
  }

  .btn-load {
    background: rgba(0, 212, 255, 0.15);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.3);
  }

  .btn-load:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.25);
  }

  .btn-load:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-unload {
    background: rgba(255, 100, 100, 0.15);
    color: #ff6464;
    border: 1px solid rgba(255, 100, 100, 0.3);
  }

  .btn-unload:hover {
    background: rgba(255, 100, 100, 0.25);
  }
</style>
