import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "MiniMaxH3VideoLLMLoader";

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function setPanel(panel, text) {
    if (panel) panel.value = String(text || "");
}

function activeModelWidget(node) {
    const provider = String(widget(node, "provider")?.value || "Ollama");
    return widget(node, provider === "LM Studio" ? "lm_studio_model" : "ollama_model");
}

function setWidgetHidden(item, hidden) {
    if (!item) return;
    if (!Object.prototype.hasOwnProperty.call(item, "__videoPromptOriginalType")) {
        item.__videoPromptOriginalType = item.type;
        item.__videoPromptOriginalComputeSize = item.computeSize;
    }
    item.hidden = hidden;
    if (hidden) {
        item.type = "converted-widget";
        item.computeSize = () => [0, -4];
        if (item.element) item.element.style.display = "none";
    } else {
        item.type = item.__videoPromptOriginalType || "combo";
        if (item.__videoPromptOriginalComputeSize) item.computeSize = item.__videoPromptOriginalComputeSize;
        else delete item.computeSize;
        if (item.element) item.element.style.display = "";
    }
}

function setProviderModelVisibility(node) {
    const provider = String(widget(node, "provider")?.value || "Ollama");
    setWidgetHidden(widget(node, "ollama_model"), provider !== "Ollama");
    setWidgetHidden(widget(node, "lm_studio_model"), provider !== "LM Studio");
    node.setDirtyCanvas(true, true);
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
        model: String(activeModelWidget(node)?.value || ""),
        node_id: String(node.id),
    };
}

async function refreshModels(node) {
    setPanel(node.__videoPromptThinking, "Refreshing installed models...");
    try {
        const payload = await post("/videoprompt_llm/models", currentConnection(node));
        const modelWidget = activeModelWidget(node);
        const models = payload.models || [];
        if (!models.length) throw new Error("The server returned no installed models.");
        modelWidget.options = { ...modelWidget.options, values: models, list: models };
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
        const originalProviderCallback = providerWidget?.callback;
        if (providerWidget) {
            providerWidget.callback = function (value) {
                originalProviderCallback?.apply(this, arguments);
                setProviderModelVisibility(node);
                refreshModels(node);
            };
        }
        const labels = {
            ollama_model: "Ollama Model",
            lm_studio_model: "LM Studio Model",
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
        setProviderModelVisibility(node);
        addButton(node, "Refresh Models", () => refreshModels(node));
        addButton(node, "Stop LLM", () => stopLLM(node));
        addButton(node, "Unload LLM", () => unloadLLM(node));
        node.__videoPromptThinking = createPanel(node, "thinking", "Thinking", 110);
        node.__videoPromptResult = createPanel(node, "result", "Result", 190);
        node.setSize([Math.max(node.size[0], 430), Math.max(node.size[1], 980)]);
        window.setTimeout(() => refreshModels(node), 0);
    },
});
