<script lang="ts">
  import { onMount } from "svelte";
  import {
    GetConfig,
    SaveConfig,
    CheckAPIHealth,
    GetVersion,
  } from "../../wailsjs/go/main/App";
  import { EventsOn } from "../../wailsjs/runtime/runtime";

  export let config = null;

  let settings = {
    theme: "dark",
    defaultModel: "ryzanstein-7b",
    defaultAgent: "@APEX",
    ryzansteinApiUrl: "http://localhost:8000",
    mcpServerUrl: "localhost:8001",
    autoLoadLastModel: true,
    enableSystemTray: false,
    minimizeToTray: false,
  };

  let isSaving = false;
  let saveMessage = "";
  let apiHealthy: boolean | null = null;
  let isCheckingHealth = false;
  let appVersion = "1.0.0";

  onMount(async () => {
    try {
      const cfg = await GetConfig();
      if (cfg) {
        settings = { ...settings, ...cfg };
      }
      appVersion = await GetVersion();
    } catch (error) {
      console.error("Failed to load config:", error);
    }

    await checkHealth();

    EventsOn("config:saved", () => {
      saveMessage = "Settings saved successfully";
      setTimeout(() => (saveMessage = ""), 3000);
    });
  });

  async function handleSave() {
    isSaving = true;
    saveMessage = "";
    try {
      await SaveConfig(settings);
      saveMessage = "Settings saved successfully";
    } catch (error) {
      saveMessage = `Error: ${error}`;
      console.error("Failed to save config:", error);
    } finally {
      isSaving = false;
      setTimeout(() => (saveMessage = ""), 3000);
    }
  }

  async function checkHealth() {
    isCheckingHealth = true;
    try {
      apiHealthy = await CheckAPIHealth();
    } catch (error) {
      apiHealthy = false;
    } finally {
      isCheckingHealth = false;
    }
  }

  function resetDefaults() {
    settings = {
      theme: "dark",
      defaultModel: "ryzanstein-7b",
      defaultAgent: "@APEX",
      ryzansteinApiUrl: "http://localhost:8000",
      mcpServerUrl: "localhost:8001",
      autoLoadLastModel: true,
      enableSystemTray: false,
      minimizeToTray: false,
    };
  }
</script>

<div class="settings-panel">
  <div class="panel-header">
    <h2>Settings</h2>
    <span class="version">v{appVersion}</span>
  </div>

  <div class="settings-content">
    <!-- Connection Settings -->
    <section class="settings-section">
      <h3>Connection</h3>

      <div class="setting-row">
        <label for="apiUrl">Ryzanstein API URL</label>
        <div class="input-with-status">
          <input
            id="apiUrl"
            type="text"
            bind:value={settings.ryzansteinApiUrl}
            placeholder="http://localhost:8000"
            class="setting-input"
          />
          <button
            class="health-btn"
            class:healthy={apiHealthy === true}
            class:unhealthy={apiHealthy === false}
            on:click={checkHealth}
            disabled={isCheckingHealth}
            title={apiHealthy === true
              ? "API is healthy"
              : apiHealthy === false
                ? "API is unreachable"
                : "Check health"}
          >
            {#if isCheckingHealth}
              ...
            {:else if apiHealthy === true}
              ✓
            {:else if apiHealthy === false}
              ✗
            {:else}
              ?
            {/if}
          </button>
        </div>
      </div>

      <div class="setting-row">
        <label for="mcpUrl">MCP Server URL</label>
        <input
          id="mcpUrl"
          type="text"
          bind:value={settings.mcpServerUrl}
          placeholder="localhost:8001"
          class="setting-input"
        />
      </div>
    </section>

    <!-- Defaults -->
    <section class="settings-section">
      <h3>Defaults</h3>

      <div class="setting-row">
        <label for="defaultModel">Default Model</label>
        <select
          id="defaultModel"
          bind:value={settings.defaultModel}
          class="setting-select"
        >
          <option value="ryzanstein-7b">Ryzanstein 7B</option>
          <option value="ryzanstein-13b">Ryzanstein 13B</option>
          <option value="bitnet-1.58b">BitNet 1.58b</option>
        </select>
      </div>

      <div class="setting-row">
        <label for="defaultAgent">Default Agent</label>
        <select
          id="defaultAgent"
          bind:value={settings.defaultAgent}
          class="setting-select"
        >
          <option value="@APEX">@APEX - Engineering</option>
          <option value="@CIPHER">@CIPHER - Security</option>
          <option value="@ARCHITECT">@ARCHITECT - Architecture</option>
          <option value="@TENSOR">@TENSOR - ML/DL</option>
          <option value="@FLUX">@FLUX - DevOps</option>
          <option value="@OMNISCIENT">@OMNISCIENT - Orchestrator</option>
        </select>
      </div>
    </section>

    <!-- Appearance -->
    <section class="settings-section">
      <h3>Appearance</h3>

      <div class="setting-row">
        <label for="theme">Theme</label>
        <select
          id="theme"
          bind:value={settings.theme}
          class="setting-select"
        >
          <option value="dark">Dark</option>
          <option value="light">Light</option>
        </select>
      </div>
    </section>

    <!-- Behavior -->
    <section class="settings-section">
      <h3>Behavior</h3>

      <div class="setting-row toggle-row">
        <label for="autoLoad">Auto-load last model on startup</label>
        <input
          id="autoLoad"
          type="checkbox"
          bind:checked={settings.autoLoadLastModel}
          class="toggle"
        />
      </div>

      <div class="setting-row toggle-row">
        <label for="sysTray">Enable system tray</label>
        <input
          id="sysTray"
          type="checkbox"
          bind:checked={settings.enableSystemTray}
          class="toggle"
        />
      </div>

      <div class="setting-row toggle-row">
        <label for="minTray">Minimize to tray</label>
        <input
          id="minTray"
          type="checkbox"
          bind:checked={settings.minimizeToTray}
          class="toggle"
        />
      </div>
    </section>
  </div>

  <!-- Actions -->
  <div class="settings-actions">
    {#if saveMessage}
      <span
        class="save-message"
        class:error={saveMessage.startsWith("Error")}
      >
        {saveMessage}
      </span>
    {/if}
    <button class="btn btn-secondary" on:click={resetDefaults}>
      Reset Defaults
    </button>
    <button class="btn btn-primary" on:click={handleSave} disabled={isSaving}>
      {isSaving ? "Saving..." : "Save Settings"}
    </button>
  </div>
</div>

<style>
  .settings-panel {
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

  .version {
    font-size: 12px;
    color: #666;
    background: rgba(0, 0, 0, 0.3);
    padding: 3px 8px;
    border-radius: 4px;
  }

  .settings-content {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .settings-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .settings-section h3 {
    margin: 0;
    font-size: 14px;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
  }

  .setting-row {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .setting-row label {
    font-size: 13px;
    color: #bbb;
  }

  .setting-input,
  .setting-select {
    padding: 8px 12px;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    color: #e0e0e0;
    font-size: 13px;
    outline: none;
    transition: border-color 0.2s;
  }

  .setting-input:focus,
  .setting-select:focus {
    border-color: rgba(0, 212, 255, 0.4);
  }

  .input-with-status {
    display: flex;
    gap: 8px;
  }

  .input-with-status .setting-input {
    flex: 1;
  }

  .health-btn {
    width: 36px;
    padding: 0;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    color: #888;
    cursor: pointer;
    font-size: 14px;
    font-weight: bold;
    transition: all 0.2s;
  }

  .health-btn.healthy {
    background: rgba(0, 200, 100, 0.15);
    border-color: rgba(0, 200, 100, 0.3);
    color: #00c864;
  }

  .health-btn.unhealthy {
    background: rgba(255, 100, 100, 0.15);
    border-color: rgba(255, 100, 100, 0.3);
    color: #ff6464;
  }

  .health-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .toggle-row {
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
  }

  .toggle {
    width: 40px;
    height: 22px;
    appearance: none;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 11px;
    position: relative;
    cursor: pointer;
    transition: all 0.3s;
    border: none;
    outline: none;
  }

  .toggle::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 18px;
    height: 18px;
    background: #888;
    border-radius: 50%;
    transition: all 0.3s;
  }

  .toggle:checked {
    background: rgba(0, 212, 255, 0.3);
  }

  .toggle:checked::after {
    left: 20px;
    background: #00d4ff;
  }

  .settings-actions {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }

  .save-message {
    font-size: 12px;
    color: #00c864;
    margin-right: auto;
  }

  .save-message.error {
    color: #ff6464;
  }

  .btn {
    padding: 8px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s;
  }

  .btn-primary {
    background: rgba(0, 212, 255, 0.2);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.3);
  }

  .btn-primary:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.3);
  }

  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-secondary {
    background: rgba(255, 255, 255, 0.05);
    color: #999;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .btn-secondary:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #ccc;
  }
</style>
