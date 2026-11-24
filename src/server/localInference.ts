import type {
  AndroidHardwareProfile,
  AttachedContextFile,
  GgufModelSpec,
  GpuPerformanceConfig,
  InferenceParams,
  PresetMode,
  QuantizationType,
} from '../types/localAi';

export interface LocalSynthesisInput {
  prompt: string;
  model: GgufModelSpec;
  quant: QuantizationType;
  hardware: AndroidHardwareProfile;
  gpuConfig: GpuPerformanceConfig;
  params: InferenceParams;
  preset: PresetMode;
  attachments: AttachedContextFile[];
  branchVariant?: number;
}

/**
 * Deterministic, domain-aware offline inference synthesizer for the local Node.js preview server.
 * On an Android device, server.py routes identically formatted SSE events directly from llama.cpp.
 */
export function buildOfflineModelResponse(input: LocalSynthesisInput): string {
  const { prompt, model, quant, hardware, gpuConfig, preset, attachments, branchVariant = 1 } = input;
  const lower = prompt.toLowerCase();
  const useThink = model.supportsThink || preset === 'deep_think';

  const attachmentSummary =
    attachments.length > 0
      ? `\n\n> **Offline Context Attached (${attachments.length} file${attachments.length > 1 ? 's' : ''})**: ${attachments
          .map((a) => `\`${a.name}\` (~${a.tokenEstimate} tokens)`)
          .join(', ')}\n`
      : '';

  const thinkBlock = useThink
    ? `<think>
Analyzing prompt locally on ${model.name} (${quant}, ${gpuConfig.backend.toUpperCase()}, ${gpuConfig.nGpuLayers}/${model.totalLayers} GPU layers, KV=${gpuConfig.kvCacheQuant}).
Hardware context: ${hardware.deviceName} · ${hardware.freeRamGb.toFixed(1)} GB free RAM · ${hardware.batteryTempC.toFixed(1)}°C.
${attachments.length > 0 ? `Parsed ${attachments.length} local file(s) into context window (${attachments.reduce((s, a) => s + a.tokenEstimate, 0)} tokens).` : 'No external attachments; answering from local quantized weights.'}
Checking edge cases, mathematical consistency, and concise structuring for mobile readability.
</think>\n\n`
    : '';

  // 1. If local files are attached, inspect and analyze their actual contents!
  if (attachments.length > 0) {
    const file = attachments[0];
    const lines = file.content.split(/\r?\n/);
    const previewLines = lines.slice(0, 14).join('\n');
    const ext = file.name.split('.').pop()?.toLowerCase() ?? 'txt';
    const lang = ['py', 'ts', 'tsx', 'js', 'kt', 'cpp', 'sql', 'json', 'md'].includes(ext)
      ? ext
      : 'text';

    return `${thinkBlock}I indexed **\`${file.name}\`** (${lines.length} lines · ~${file.tokenEstimate} tokens) directly into the offline \`${model.name}\` context window.${attachmentSummary}

### Key Findings from \`${file.name}\`
- **Line & Symbol Footprint**: ${lines.length} lines parsed with zero cloud upload (100% on-device).
- **Prompt Focus**: Addressing *"${prompt.slice(0, 110)}${prompt.length > 110 ? '...' : ''}"* using **${preset.replace('_', ' ')}** sampling (\`T=${input.params.temperature}\`, \`KV=${gpuConfig.kvCacheQuant}\`).

\`\`\`${lang}
${previewLines}${lines.length > 14 ? `\n// ... (${lines.length - 14} more lines in local context)` : ''}
\`\`\`

If you want me to refactor a specific function, generate unit tests, or summarize another section of \`${file.name}\`, point to the symbol or line range and I will run it directly on ${gpuConfig.backend.toUpperCase()}.`;
  }

  // 2. Math / physics / equation / formula prompts -> render real KaTeX
  if (/\b(math|equation|integral|derivative|matrix|formula|attention|softmax|prove|calculate|quadratic)\b/.test(lower)) {
    const altNote =
      branchVariant > 1
        ? '\n\n*Branch ' + branchVariant + ' formulation emphasizes numerical stability via log-sum-exp scaling.*'
        : '';
    return `${thinkBlock}Here is the exact derivation computed offline by **${model.name}** (\`${quant}\`):

### Scaled Dot-Product Attention & Quantized Memory Bound

Given query $Q$, key $K$, and value $V$ matrices of head dimension $d_k$:

$$\\operatorname{Attention}(Q, K, V) = \\operatorname{softmax}\\!\\left(\\frac{Q K^{\\top}}{\\sqrt{d_k}}\\right) V$$

To keep exponentiation numerically stable in \`FP16\` / \`Q8_0\` mobile Vulkan shaders, subtract the row maximum $m_i = \\max_j s_{ij}$ before exponentiating:

$$\\operatorname{softmax}(s_i)_j = \\frac{\\exp(s_{ij} - m_i)}{\\sum_{k=1}^{C} \\exp(s_{ik} - m_i)}$$

With **Flash Attention** (${gpuConfig.flashAttention ? 'enabled' : 'disabled'}) and **\`${gpuConfig.kvCacheQuant}\`** KV-cache quantization, peak attention memory scales as $\\mathcal{O}(C)$ instead of $\\mathcal{O}(C^2)$ across your ${input.params.contextLength.toLocaleString()}-token context window.${altNote}`;
  }

  // 3. Code / Kotlin / Python / C++ / SQL / API prompts -> render real syntax-highlighted code block
  if (/\b(code|kotlin|python|sqlite|fts5|vulkan|opencl|llama|function|script|sort|algorithm|bug|refactor|regex|api)\b/.test(lower) || preset === 'coder') {
    if (lower.includes('kotlin') || lower.includes('android')) {
      return `${thinkBlock}Here is a production Kotlin snippet tuned for **${hardware.deviceName}** (${hardware.abi}):

\`\`\`kotlin
fun readAndroidMemoryHeadroom(context: Context): Pair<Double, Double> {
    val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val info = ActivityManager.MemoryInfo()
    am.getMemoryInfo(info)
    val gb = 1024.0 * 1024.0 * 1024.0
    val totalGb = info.totalMem / gb
    val availGb = info.availMem / gb
    return Pair(totalGb, availGb)
}
\`\`\`

### Implementation Notes
- Keeps a **1.15 GB** safety margin before recommending \`Smooth on Your Phone (GPU Ready)\`.
- Pair with \`mmap = ${gpuConfig.mmap}\` and \`n_threads = ${gpuConfig.nThreads}\` (pinned to Prime/Performance cores) to avoid efficiency-core stalls during decode.`;
    }

    return `${thinkBlock}Here is a clean, offline-ready Python implementation generated by **${model.name}** (\`${quant}\`):

\`\`\`python
from dataclasses import dataclass
import hashlib
from pathlib import Path

@dataclass(frozen=True)
class GGUFCheck:
    path: Path
    size_mb: float
    sha256: str
    valid_magic: bool

def verify_gguf_artifact(file_path: str) -> GGUFCheck:
    path = Path(file_path)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        header = fh.read(4)
        digest.update(header)
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return GGUFCheck(
        path=path,
        size_mb=round(path.stat().st_size / (1024 * 1024), 2),
        sha256=digest.hexdigest(),
        valid_magic=(header == b"GGUF"),
    )
\`\`\`

### Complexity & On-Device Notes
- **Streaming I/O**: Reads 1 MB aligned buffers so verifying a 4.9 GB Level 3 model stays under **4 MB** peak heap usage.
- **Zero-Copy Safe**: Validates the \`GGUF\` 4-byte magic before handing the descriptor to \`libllama.so\`.${branchVariant > 1 ? `\n- **Regenerated Variant #${branchVariant}**: Optimized for strict zero-allocation streaming.` : ''}`;
  }

  // 4. General crisp mobile assistant answer
  const variantLabel =
    branchVariant > 1 ? ` (Alternative Branch #${branchVariant})` : '';

  return `${thinkBlock}Running 100% offline on **${model.name}** (\`L${model.level} · ${quant}\`) via **${gpuConfig.backend.toUpperCase()}**${variantLabel}.

Regarding **"${prompt.trim()}"**:

1. **Direct Answer**: Your query is processed locally with zero network round-trips, using \`${gpuConfig.nGpuLayers}/${model.totalLayers}\` GPU-offloaded layers, \`${gpuConfig.kvCacheQuant}\` KV-cache, and \`${gpuConfig.nThreads}\` pinned performance-core threads on **${hardware.deviceName}**.
2. **Recommended Local Workflow**:
   - Use **Fast** preset for sub-200ms quick replies and translations.
   - Switch to **Deep Think** (or load a Level 3/4 \`DeepSeek-R1-Distill\` model) when you want step-by-step \`<think>\` chain-of-thought verification.
   - Attach local \`.py\`, \`.kt\`, \`.md\`, or \`.pdf\` files via the **\`+\`** composer button to inject them directly into your ${input.params.contextLength.toLocaleString()}-token offline context.

Ask for a Kotlin/Python code example, a KaTeX math derivation, or attach a local document to inspect it offline.`;
}
