import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "MiniMaxH3VideoLLMLoader";

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function setPanel(panel, text) {
    if (panel) panel.value = String(text || "");
}

function setModelLabel(node) {
    const modelWidget = widget(node, "model");
    const provider = String(widget(node, "provider")?.value || "Ollama");
    if (modelWidget) modelWidget.label = provider === "LM Studio" ? "LM Studio Model" : "Ollama Model";
}

function createPanel(node, name, label, height) {
    const element = document.createElement("textarea");
    element.readOnly = true;
    element.placeholder = label;
    element.style.cssText = [
        "width: 100%", `height: ${height}px`, "resize: vertical", "box-sizing: border-box",
        "background: #171a1f", "color: #dce7e2", "border: 1px solid #46514d",
        "border-radius: 5px", "padding: 7px", "font: 12px/1.4 monospace", "white-space: pre-wrap",
    ].join(";");
    const domWidget = node.addDOMWidget(`videoprompt_${name}`, label, element, {
        serialize: false, hideOnZoom: false, getMinHeight: () => height + 24,
    });
    domWidget.computeSize = () => [node.size[0], height + 24];
    return element;
}

async function post(path, body) {
    const response = await api.fetchApi(path, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
}

function currentConnection(node) {
    return {
        provider: String(widget(node, "provider")?.value || "Ollama"),
        server_url: String(widget(node, "server_url")?.value || ""),
        model: String(widget(node, "model")?.value || ""),
        node_id: String(node.id),
    };
}

async function refreshModels(node) {
    setPanel(node.__videoPromptThinking, "Refreshing installed models...");
    try {
        const payload = await post("/videoprompt_llm/models", currentConnection(node));
        const modelWidget = widget(node, "model");
        const models = payload.models || [];
        if (!models.length) throw new Error("The server returned no installed models.");
        modelWidget.options.values = models;
        if (!models.includes(String(modelWidget.value))) modelWidget.value = models[0];
        setPanel(node.__videoPromptThinking, `Detected ${models.length} model(s). Select a vision-capable model.`);
        node.setDirtyCanvas(true, true);
    } catch (error) {
        setPanel(node.__videoPromptThinking, `Refresh Models error: ${error.message}`);
    }
}

async function stopLLM(node) {
    try {
        const payload = await post("/videoprompt_llm/stop", currentConnection(node));
        setPanel(node.__videoPromptThinking, payload.stopped_active_request
            ? "Stop requested for the active LLM call." : "No active LLM call was found.");
    } catch (error) {
        setPanel(node.__videoPromptThinking, `Stop LLM error: ${error.message}`);
    }
}

async function unloadLLM(node) {
    setPanel(node.__videoPromptThinking, "Requesting model unload...");
    try {
        const payload = await post("/videoprompt_llm/unload", currentConnection(node));
        setPanel(node.__videoPromptThinking, payload.message || "Model unloaded.");
    } catch (error) {
        setPanel(node.__videoPromptThinking, `Unload LLM error: ${error.message}`);
    }
}

function addButton(node, name, callback) {
    const button = node.addWidget("button", name, name, callback, { serialize: false });
    button.serializeValue = () => undefined;
}

api.addEventListener("videoprompt_llm_progress", (event) => {
    const payload = event.detail || event;
    const node = app.graph?.getNodeById?.(payload.node_id);
    if (!node || node.comfyClass !== NODE_NAME) return;
    if (Object.prototype.hasOwnProperty.call(payload, "thinking")) setPanel(node.__videoPromptThinking, payload.thinking);
    if (Object.prototype.hasOwnProperty.call(payload, "result") && payload.result) setPanel(node.__videoPromptResult, payload.result);
    node.setDirtyCanvas(true, true);
});

app.registerExtension({
    name: "DJwanho.VideoPromptLLM",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            originalExecuted?.apply(this, arguments);
            const thinking = Array.isArray(message?.thinking) ? message.thinking[0] : message?.thinking;
            const result = Array.isArray(message?.text) ? message.text[0] : message?.text;
            if (thinking !== undefined) setPanel(this.__videoPromptThinking, thinking);
            if (result !== undefined) setPanel(this.__videoPromptResult, result);
        };
    },
    nodeCreated(node) {
        if (node.comfyClass !== NODE_NAME) return;
        const providerWidget = widget(node, "provider");
        const serverWidget = widget(node, "server_url");
        const originalProviderCallback = providerWidget?.callback;
        if (providerWidget) {
            providerWidget.callback = function (value) {
                originalProviderCallback?.apply(this, arguments);
                if (serverWidget) serverWidget.value = value === "LM Studio"
                    ? "http://127.0.0.1:1234" : "http://127.0.0.1:11434";
                setModelLabel(node);
            };
        }
        const labels = {
            custom_system_prompt: "System Prompt / Custom",
            prompt: "Prompt",
            model_after_run: "Model After Run",
            unload_comfyui_models: "Unload ComfyUI Models Setting",
            max_analysis_frames: "Max Analysis Frames",
            analysis_resolution: "Analysis Image Resolution",
        };
        for (const [name, label] of Object.entries(labels)) {
            const item = widget(node, name);
            if (item) item.label = label;
        }
        setModelLabel(node);
        addButton(node, "Refresh Models", () => refreshModels(node));
        addButton(node, "Stop LLM", () => stopLLM(node));
        addButton(node, "Unload LLM", () => unloadLLM(node));
        node.__videoPromptThinking = createPanel(node, "thinking", "Thinking", 110);
        node.__videoPromptResult = createPanel(node, "result", "Result", 190);
        node.setSize([Math.max(node.size[0], 430), Math.max(node.size[1], 980)]);
    },
});
