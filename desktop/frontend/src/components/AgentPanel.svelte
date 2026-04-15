<script lang="ts">
  import { onMount } from "svelte";
  import { ListAgents, InvokeAgent } from "../../wailsjs/go/main/App";

  export let config = null;

  // Elite Agent Collective agent metadata
  const agentMetadata: Record<
    string,
    { name: string; tier: number; philosophy: string; emoji: string }
  > = {
    "@APEX": {
      name: "Elite CS Engineering",
      tier: 1,
      philosophy: "Every problem has an elegant solution.",
      emoji: "💻",
    },
    "@CIPHER": {
      name: "Cryptography & Security",
      tier: 1,
      philosophy: "Security is a foundation, not a feature.",
      emoji: "🔐",
    },
    "@ARCHITECT": {
      name: "Systems Architecture",
      tier: 1,
      philosophy: "Making complexity manageable.",
      emoji: "🏛️",
    },
    "@AXIOM": {
      name: "Mathematics & Proofs",
      tier: 1,
      philosophy: "From axioms flow certainty.",
      emoji: "📐",
    },
    "@VELOCITY": {
      name: "Performance Optimization",
      tier: 1,
      philosophy: "The fastest code is code that doesn't run.",
      emoji: "⚡",
    },
    "@QUANTUM": {
      name: "Quantum Computing",
      tier: 2,
      philosophy: "Superposition is power.",
      emoji: "⚛️",
    },
    "@TENSOR": {
      name: "Machine Learning",
      tier: 2,
      philosophy: "Intelligence from architecture + data.",
      emoji: "🧠",
    },
    "@FORTRESS": {
      name: "Penetration Testing",
      tier: 2,
      philosophy: "Think like the attacker.",
      emoji: "🛡️",
    },
    "@NEURAL": {
      name: "AGI Research",
      tier: 2,
      philosophy: "Synthesis of specialized capabilities.",
      emoji: "🔬",
    },
    "@CRYPTO": {
      name: "Blockchain Systems",
      tier: 2,
      philosophy: "Trust is computed and verified.",
      emoji: "⛓️",
    },
    "@FLUX": {
      name: "DevOps & Infrastructure",
      tier: 2,
      philosophy: "Infrastructure is code.",
      emoji: "🔄",
    },
    "@PRISM": {
      name: "Data Science",
      tier: 2,
      philosophy: "Ask the right questions.",
      emoji: "📊",
    },
    "@SYNAPSE": {
      name: "API & Integration",
      tier: 2,
      philosophy: "Systems are their connections.",
      emoji: "🔗",
    },
    "@CORE": {
      name: "Low-Level Systems",
      tier: 2,
      philosophy: "Every instruction counts.",
      emoji: "⚙️",
    },
    "@ECLIPSE": {
      name: "Testing & Verification",
      tier: 2,
      philosophy: "Untested code is broken code.",
      emoji: "🧪",
    },
    "@NEXUS": {
      name: "Cross-Domain Synthesis",
      tier: 3,
      philosophy: "Ideas at the intersection of domains.",
      emoji: "🌐",
    },
    "@GENESIS": {
      name: "Innovation & Discovery",
      tier: 3,
      philosophy: "Discoveries are revelations.",
      emoji: "💡",
    },
    "@OMNISCIENT": {
      name: "Meta-Orchestrator",
      tier: 4,
      philosophy: "Collective intelligence exceeds the sum.",
      emoji: "👁️",
    },
  };

  let agents: string[] = [];
  let selectedAgent: string | null = null;
  let filterTier: number | null = null;
  let searchQuery = "";
  let isLoading = true;

  onMount(async () => {
    try {
      agents = await ListAgents();
    } catch (error) {
      console.error("Failed to list agents:", error);
    } finally {
      isLoading = false;
    }
  });

  function getAgentMeta(codename: string) {
    return (
      agentMetadata[codename] || {
        name: codename.replace("@", ""),
        tier: 0,
        philosophy: "",
        emoji: "🤖",
      }
    );
  }

  function getTierLabel(tier: number): string {
    switch (tier) {
      case 1:
        return "Foundational";
      case 2:
        return "Specialist";
      case 3:
        return "Innovator";
      case 4:
        return "Meta";
      default:
        return "Unknown";
    }
  }

  function getTierColor(tier: number): string {
    switch (tier) {
      case 1:
        return "#00d4ff";
      case 2:
        return "#7c3aed";
      case 3:
        return "#f59e0b";
      case 4:
        return "#ef4444";
      default:
        return "#888";
    }
  }

  $: filteredAgents = agents.filter((a) => {
    const meta = getAgentMeta(a);
    const matchesTier = filterTier === null || meta.tier === filterTier;
    const matchesSearch =
      searchQuery === "" ||
      a.toLowerCase().includes(searchQuery.toLowerCase()) ||
      meta.name.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesTier && matchesSearch;
  });
</script>

<div class="agent-panel">
  <div class="panel-header">
    <h2>Elite Agent Collective</h2>
    <span class="agent-count">{agents.length} agents</span>
  </div>

  <div class="filters">
    <input
      type="text"
      placeholder="Search agents..."
      bind:value={searchQuery}
      class="search-input"
    />
    <div class="tier-filters">
      <button
        class="tier-btn"
        class:active={filterTier === null}
        on:click={() => (filterTier = null)}
      >
        All
      </button>
      {#each [1, 2, 3, 4] as tier}
        <button
          class="tier-btn"
          class:active={filterTier === tier}
          on:click={() => (filterTier = filterTier === tier ? null : tier)}
          style="--tier-color: {getTierColor(tier)}"
        >
          T{tier}
        </button>
      {/each}
    </div>
  </div>

  {#if isLoading}
    <div class="loading-state">
      <p>Loading agents...</p>
    </div>
  {:else}
    <div class="agent-grid">
      {#each filteredAgents as agent (agent)}
        {@const meta = getAgentMeta(agent)}
        <button
          class="agent-card"
          class:selected={selectedAgent === agent}
          on:click={() => (selectedAgent = selectedAgent === agent ? null : agent)}
        >
          <div class="agent-emoji">{meta.emoji}</div>
          <div class="agent-info">
            <div class="agent-name">{agent}</div>
            <div class="agent-role">{meta.name}</div>
            <div
              class="agent-tier"
              style="color: {getTierColor(meta.tier)}"
            >
              Tier {meta.tier} — {getTierLabel(meta.tier)}
            </div>
          </div>
        </button>
      {/each}
    </div>

    {#if selectedAgent}
      {@const meta = getAgentMeta(selectedAgent)}
      <div class="agent-detail">
        <div class="detail-header">
          <span class="detail-emoji">{meta.emoji}</span>
          <div>
            <h3>{selectedAgent}</h3>
            <p class="detail-role">{meta.name}</p>
          </div>
        </div>
        {#if meta.philosophy}
          <p class="detail-philosophy">"{meta.philosophy}"</p>
        {/if}
        <p class="detail-hint">
          Use <code>{selectedAgent}</code> in the chat panel to invoke this agent.
        </p>
      </div>
    {/if}
  {/if}
</div>

<style>
  .agent-panel {
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

  .agent-count {
    font-size: 12px;
    color: #888;
    background: rgba(0, 0, 0, 0.3);
    padding: 4px 10px;
    border-radius: 12px;
  }

  .filters {
    display: flex;
    gap: 12px;
    align-items: center;
  }

  .search-input {
    flex: 1;
    padding: 8px 12px;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    color: #e0e0e0;
    font-size: 13px;
    outline: none;
  }

  .search-input:focus {
    border-color: rgba(0, 212, 255, 0.4);
  }

  .tier-filters {
    display: flex;
    gap: 4px;
  }

  .tier-btn {
    padding: 6px 12px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    color: #888;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s;
  }

  .tier-btn:hover {
    background: rgba(255, 255, 255, 0.05);
    color: #ccc;
  }

  .tier-btn.active {
    background: rgba(0, 212, 255, 0.15);
    border-color: rgba(0, 212, 255, 0.3);
    color: var(--tier-color, #00d4ff);
  }

  .loading-state {
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 1;
    color: #888;
  }

  .agent-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 8px;
    overflow-y: auto;
    flex: 1;
    padding: 4px;
  }

  .agent-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: rgba(0, 0, 0, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s;
    text-align: left;
    color: #e0e0e0;
  }

  .agent-card:hover {
    background: rgba(0, 0, 0, 0.25);
    border-color: rgba(255, 255, 255, 0.12);
  }

  .agent-card.selected {
    background: rgba(0, 212, 255, 0.08);
    border-color: rgba(0, 212, 255, 0.3);
  }

  .agent-emoji {
    font-size: 24px;
    width: 40px;
    text-align: center;
    flex-shrink: 0;
  }

  .agent-info {
    flex: 1;
    min-width: 0;
  }

  .agent-name {
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .agent-role {
    font-size: 12px;
    color: #999;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .agent-tier {
    font-size: 11px;
    margin-top: 2px;
  }

  .agent-detail {
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 16px;
    margin-top: 8px;
  }

  .detail-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }

  .detail-emoji {
    font-size: 32px;
  }

  .detail-header h3 {
    margin: 0;
    font-size: 18px;
  }

  .detail-role {
    margin: 2px 0 0;
    font-size: 13px;
    color: #999;
  }

  .detail-philosophy {
    font-style: italic;
    color: #aaa;
    font-size: 13px;
    margin: 8px 0;
  }

  .detail-hint {
    font-size: 12px;
    color: #666;
  }

  .detail-hint code {
    background: rgba(0, 212, 255, 0.1);
    color: #00d4ff;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 12px;
  }
</style>
