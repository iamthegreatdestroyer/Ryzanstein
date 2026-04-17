import * as vscode from "vscode";
import { ExtensionContext } from "vscode";
import { RyzansteinClient } from "../client/RyzansteinClient";
import { MCPClient } from "../client/MCPClient";
import { ChatWebviewProvider } from "../providers/ChatWebviewProvider";

export class CommandHandler {
  constructor(
    context: ExtensionContext,
    private ryzansteinClient: RyzansteinClient,
    private mcpClient: MCPClient
  ) {
    this.registerCommands(context);
  }

  private registerCommands(context: ExtensionContext) {
    // Open Chat Command
    const openChatCommand = vscode.commands.registerCommand(
      "ryzanstein.openChat",
      async () => {
        const panel = vscode.window.createWebviewPanel(
          "ryzansteinChat",
          "Ryzanstein Chat",
          vscode.ViewColumn.Beside,
          { enableScripts: true }
        );
        panel.webview.html = this.getChatWebviewContent();
        panel.webview.onDidReceiveMessage(async (message) => {
          if (message.type === "chat") {
            try {
              const result = await this.ryzansteinClient.infer(message.message);
              panel.webview.postMessage({ type: "response", message: result });
            } catch (err: any) {
              panel.webview.postMessage({
                type: "response",
                message: `Error: ${err.message}`,
              });
            }
          }
        });
      }
    );

    // Select Agent Command
    const selectAgentCommand = vscode.commands.registerCommand(
      "ryzanstein.selectAgent",
      async () => {
        const agents = await this.ryzansteinClient.listAgents();
        const agentNames = agents.map((a) => a.name);
        const selected = await vscode.window.showQuickPick(agentNames);
        if (selected) {
          vscode.workspace
            .getConfiguration("ryzanstein")
            .update("selectedAgent", selected);
          vscode.window.showInformationMessage(`✓ Selected agent: ${selected}`);
        }
      }
    );

    // Load Model Command
    const loadModelCommand = vscode.commands.registerCommand(
      "ryzanstein.loadModel",
      async (modelId: string) => {
        try {
          await this.ryzansteinClient.loadModel(modelId);
          vscode.window.showInformationMessage(`✓ Loaded model: ${modelId}`);
        } catch (error) {
          vscode.window.showErrorMessage(
            `Failed to load model: ${
              error instanceof Error ? error.message : "Unknown error"
            }`
          );
        }
      }
    );

    // Generate Code Command
    const generateCodeCommand = vscode.commands.registerCommand(
      "ryzanstein.generateCode",
      async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
          vscode.window.showErrorMessage("No active editor");
          return;
        }

        const prompt = await vscode.window.showInputBox({
          prompt: "Enter your code generation prompt",
          placeHolder: "e.g., Generate a TypeScript function to...",
        });

        if (!prompt) return;

        try {
          const code = await this.ryzansteinClient.generateCode(prompt);
          editor.edit((editBuilder) => {
            editBuilder.insert(editor.selection.active, code);
          });
        } catch (error) {
          vscode.window.showErrorMessage(
            `Code generation failed: ${
              error instanceof Error ? error.message : "Unknown error"
            }`
          );
        }
      }
    );

const inferCommand = vscode.commands.registerCommand(
          "ryzanstein.infer",
          async () => {
            const prompt = await vscode.window.showInputBox({
              prompt: "Enter inference prompt",
              placeHolder: "Ask Ryzanstein...",
            });
            if (!prompt) return;
            try {
              const result = await this.ryzansteinClient.infer(prompt);
              vscode.window.showInformationMessage(`Ryzanstein: ${result}`);
            } catch (error) {
              vscode.window.showErrorMessage(
                `Inference failed: ${
                  error instanceof Error ? error.message : "Unknown error"
                }`
              );
            }
          }
        );

        context.subscriptions.push(
          openChatCommand,
          selectAgentCommand,
          loadModelCommand,
          generateCodeCommand,
          inferCommand
        );
      }

      private getChatWebviewContent(): string {
        return `<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Ryzanstein Chat</title>
      <style>
        body { font-family: var(--vscode-font-family); padding: 10px; background: var(--vscode-editor-background); color: var(--vscode-editor-foreground); }
        #messages { height: 70vh; overflow-y: auto; border: 1px solid var(--vscode-panel-border); padding: 10px; margin-bottom: 10px; }
        #input { width: 80%; padding: 8px; background: var(--vscode-input-background); color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border); }
        button { padding: 8px 16px; background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; cursor: pointer; }
      </style>
    </head>
    <body>
      <h2>Ryzanstein Chat</h2>
      <div id="messages"></div>
      <input id="input" type="text" placeholder="Type a message..." />
      <button onclick="sendMessage()">Send</button>
      <script>
        const vscode = acquireVsCodeApi();
        function sendMessage() {
          const input = document.getElementById('input');
          const msg = input.value.trim();
          if (!msg) return;
          appendMessage('You', msg);
          vscode.postMessage({ type: 'chat', message: msg });
          input.value = '';
        }
        function appendMessage(sender, text) {
          const div = document.getElementById('messages');
          const p = document.createElement('p');
          p.innerHTML = '<strong>' + sender + ':</strong> ' + text;
          div.appendChild(p);
          div.scrollTop = div.scrollHeight;
        }
        document.getElementById('input').addEventListener('keypress', (e) => {
          if (e.key === 'Enter') sendMessage();
        });
        window.addEventListener('message', (event) => {
          const data = event.data;
          if (data.type === 'response') appendMessage('Ryzanstein', data.message);
        });
      </script>
    </body>
    </html>`;
  }
}
