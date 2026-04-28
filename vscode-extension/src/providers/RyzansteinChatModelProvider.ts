import * as vscode from "vscode";
import { RyzansteinClient } from "../client/RyzansteinClient";

export interface RyzansteinModel {
  id: string;
  vendor: string;
  family: string;
  name: string;
  label: string;
  version: string;
  maxInputTokens: number;
  maxOutputTokens: number;
}

const MODELS: RyzansteinModel[] = [
  {
    id: "ryzanstein",
    vendor: "ryzanstein",
    family: "ryzanstein",
    name: "Ryzanstein",
    label: "Ryzanstein",
    version: "1.0.0",
    maxInputTokens: 4096,
    maxOutputTokens: 2048,
  },
  {
    id: "ryzanstein-apex",
    vendor: "ryzanstein",
    family: "ryzanstein-agent",
    name: "APEX",
    label: "Ryzanstein (@APEX - Elite CS Engineering)",
    version: "1.0.0",
    maxInputTokens: 4096,
    maxOutputTokens: 2048,
  },
  {
    id: "ryzanstein-architect",
    vendor: "ryzanstein",
    family: "ryzanstein-agent",
    name: "ARCHITECT",
    label: "Ryzanstein (@ARCHITECT - Systems Design)",
    version: "1.0.0",
    maxInputTokens: 4096,
    maxOutputTokens: 2048,
  },
  {
    id: "ryzanstein-tensor",
    vendor: "ryzanstein",
    family: "ryzanstein-agent",
    name: "TENSOR",
    label: "Ryzanstein (@TENSOR - Machine Learning)",
    version: "1.0.0",
    maxInputTokens: 4096,
    maxOutputTokens: 2048,
  },
  {
    id: "ryzanstein-cipher",
    vendor: "ryzanstein",
    family: "ryzanstein-agent",
    name: "CIPHER",
    label: "Ryzanstein (@CIPHER - Security)",
    version: "1.0.0",
    maxInputTokens: 4096,
    maxOutputTokens: 2048,
  },
];

export class RyzansteinChatModelProvider {
  constructor(private ryzansteinClient: RyzansteinClient) {}

  getModels(): RyzansteinModel[] {
    return MODELS;
  }

  getModel(id: string): RyzansteinModel | undefined {
    return MODELS.find((m) => m.id === id);
  }
}

export class RyzansteinChatResponseProvider {
  constructor(private ryzansteinClient: RyzansteinClient) {}

  private agentFromModelId(modelId: string): string {
    if (modelId.includes("apex")) return "APEX";
    if (modelId.includes("architect")) return "ARCHITECT";
    if (modelId.includes("tensor")) return "TENSOR";
    if (modelId.includes("cipher")) return "CIPHER";
    return "default";
  }

  async chat(
    userInput: string,
    modelId: string,
    token?: vscode.CancellationToken
  ): Promise<{ text: string; traceId?: string }> {
    const agent = this.agentFromModelId(modelId);
    const response = await this.ryzansteinClient.chat(userInput, agent);
    return { text: response.response, traceId: response.traceId };
  }
}
