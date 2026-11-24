package com.localcore.ai

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.GLES20
import android.os.BatteryManager
import android.os.Build
import android.os.Environment
import android.os.PowerManager
import android.os.StatFs
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.webkit.JavascriptInterface
import android.webkit.WebView
import org.json.JSONObject
import java.io.File
import java.util.Locale

/**
 * LocalCore AI — Kotlin Android Native Bridge
 * Exposes real Android RAM telemetry, GLES/Vulkan GPU renderer, battery/SoC thermal status,
 * Storage Access Framework (SAF) zero-copy .gguf file picker, ForegroundService wake-lock &
 * download notifications, native streaming TTS, haptics, and onTrimMemory() LMK callbacks.
 */
class AndroidHardwareBridge(
    private val activity: MainActivity,
    private val webView: WebView
) : TextToSpeech.OnInitListener {

    private val appContext: Context = activity.applicationContext
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private val cachedGpuRenderer: String by lazy { queryGlesRenderer() }

    init {
        tts = TextToSpeech(appContext, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.language = Locale.US
            ttsReady = true
        }
    }

    /**
     * Returns real-time hardware telemetry as JSON to the WebView & Python GGUF engine.
     */
    @JavascriptInterface
    fun getHardwareTelemetryJson(): String {
        val am = appContext.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)

        val totalRamGb = memInfo.totalMem.toDouble() / (1024.0 * 1024.0 * 1024.0)
        val freeRamGb = memInfo.availMem.toDouble() / (1024.0 * 1024.0 * 1024.0)

        val statFs = StatFs(appContext.filesDir.absolutePath)
        val storageFreeGb =
            (statFs.availableBlocksLong * statFs.blockSizeLong).toDouble() / (1024.0 * 1024.0 * 1024.0)

        val batteryTempC = readBatteryTemperatureC()
        val thermalStatus = readThermalStatus()
        val gpuRenderer = cachedGpuRenderer
        val gpuVendor = when {
            gpuRenderer.contains("Adreno", ignoreCase = true) -> "Adreno"
            gpuRenderer.contains("Mali", ignoreCase = true) ||
                gpuRenderer.contains("Immortalis", ignoreCase = true) -> "Mali"
            gpuRenderer.contains("Xclipse", ignoreCase = true) -> "Xclipse"
            else -> "Generic"
        }

        val vulkanSupported = appContext.packageManager.hasSystemFeature(
            PackageManager.FEATURE_VULKAN_HARDWARE_VERSION
        )
        val openclSupported = File("/system/vendor/lib64/libOpenCL.so").exists() ||
            File("/vendor/lib64/libOpenCL.so").exists() ||
            gpuVendor == "Adreno" || gpuVendor == "Mali"

        val totalCores = Runtime.getRuntime().availableProcessors()
        val primeCores = (totalCores - 2).coerceAtLeast(2).coerceAtMost(6)

        return JSONObject().apply {
            put("id", "android-live-device")
            put("deviceName", "${Build.MANUFACTURER} ${Build.MODEL}")
            put("socName", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE)
            put("gpuRenderer", gpuRenderer)
            put("gpuVendor", gpuVendor)
            put("vulkanSupported", vulkanSupported)
            put("openclSupported", openclSupported)
            put("totalRamGb", String.format(Locale.US, "%.2f", totalRamGb).toDouble())
            put("freeRamGb", String.format(Locale.US, "%.2f", freeRamGb).toDouble())
            put("lowMemory", memInfo.lowMemory)
            put("storageFreeGb", String.format(Locale.US, "%.2f", storageFreeGb).toDouble())
            put("totalCores", totalCores)
            put("primePerformanceCores", primeCores)
            put("batteryTempC", batteryTempC)
            put("thermalStatus", thermalStatus)
            put("androidApi", Build.VERSION.SDK_INT)
            put("abi", Build.SUPPORTED_ABIS.firstOrNull() ?: "arm64-v8a")
            put("filesDir", appContext.filesDir.absolutePath)
            put("isLiveBridge", true)
        }.toString()
    }

    /**
     * Launches Android Storage Access Framework (SAF) document picker for .gguf files
     * without copying multi-GB files into app private storage.
     */
    @JavascriptInterface
    fun openSafGgufPicker() {
        activity.runOnUiThread {
            activity.launchSafGgufPicker()
        }
    }

    /**
     * Starts or updates the Android ForegroundService to keep downloads and long generations
     * alive with a PARTIAL_WAKE_LOCK and live notification progress bar.
     */
    @JavascriptInterface
    fun startForegroundTask(title: String, detail: String, progressPercent: Int) {
        val intent = Intent(appContext, ModelDownloadService::class.java).apply {
            action = ModelDownloadService.ACTION_UPDATE
            putExtra(ModelDownloadService.EXTRA_TITLE, title)
            putExtra(ModelDownloadService.EXTRA_DETAIL, detail)
            putExtra(ModelDownloadService.EXTRA_PROGRESS, progressPercent)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            appContext.startForegroundService(intent)
        } else {
            appContext.startService(intent)
        }
    }

    @JavascriptInterface
    fun stopForegroundTask() {
        val intent = Intent(appContext, ModelDownloadService::class.java).apply {
            action = ModelDownloadService.ACTION_STOP
        }
        appContext.startService(intent)
    }

    /**
     * Native Android TextToSpeech streaming hook (replaces unreliable WebView window.speechSynthesis).
     */
    @JavascriptInterface
    fun speakNative(text: String, flushQueue: Boolean) {
        if (!ttsReady) return
        val mode = if (flushQueue) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
        tts?.speak(text, mode, null, "localcore_tts_${System.currentTimeMillis()}")
    }

    @JavascriptInterface
    fun stopSpeaking() {
        tts?.stop()
    }

    /**
     * Triggers subtle Android hardware haptic feedback for touch actions and token abort.
     */
    @JavascriptInterface
    fun triggerHaptic(type: String) {
        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (appContext.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            appContext.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
        if (!vibrator.hasVibrator()) return

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val effect = when (type) {
                "heavy" -> VibrationEffect.createOneShot(32, VibrationEffect.DEFAULT_AMPLITUDE)
                "success" -> VibrationEffect.createWaveform(longArrayOf(0, 14, 30, 18), -1)
                else -> VibrationEffect.createOneShot(12, 90)
            }
            vibrator.vibrate(effect)
        }
    }

    /**
     * Dispatches Android onTrimMemory() pressure callbacks to the WebView & Python engine
     * so KV-cache or model weights are gracefully unloaded before LMK kills the process.
     */
    fun notifyTrimMemory(level: Int) {
        val js = "window.dispatchEvent(new CustomEvent('android-trim-memory', { detail: { level: $level } }));"
        webView.post { webView.evaluateJavascript(js, null) }
    }

    fun notifySafFilePicked(uriString: String, displayName: String, sizeBytes: Long) {
        val payload = JSONObject().apply {
            put("uri", uriString)
            put("name", displayName)
            put("sizeBytes", sizeBytes)
        }
        val js = "window.dispatchEvent(new CustomEvent('android-saf-gguf-picked', { detail: $payload }));"
        webView.post { webView.evaluateJavascript(js, null) }
    }

    fun release() {
        tts?.stop()
        tts?.shutdown()
        tts = null
    }

    private fun readBatteryTemperatureC(): Double {
        val intent = appContext.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tenths = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 350) ?: 350
        return tenths / 10.0
    }

    private fun readThermalStatus(): String {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return "nominal"
        val pm = appContext.getSystemService(Context.POWER_SERVICE) as PowerManager
        return when (pm.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE,
            PowerManager.THERMAL_STATUS_LIGHT -> "nominal"
            PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
            PowerManager.THERMAL_STATUS_SEVERE -> "severe"
            else -> "critical"
        }
    }

    private fun queryGlesRenderer(): String {
        return try {
            val display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            val version = IntArray(2)
            EGL14.eglInitialize(display, version, 0, version, 1)
            val attribList = intArrayOf(
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_NONE
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            EGL14.eglChooseConfig(display, attribList, 0, configs, 0, 1, numConfigs, 0)
            val context = EGL14.eglCreateContext(
                display, configs[0], EGL14.EGL_NO_CONTEXT,
                intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE), 0
            )
            val surface = EGL14.eglCreatePbufferSurface(
                display, configs[0],
                intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0
            )
            EGL14.eglMakeCurrent(display, surface, surface, context)
            val renderer = GLES20.glGetString(GLES20.GL_RENDERER) ?: "Unknown Mobile GPU"
            EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            EGL14.eglDestroySurface(display, surface)
            EGL14.eglDestroyContext(display, context)
            EGL14.eglTerminate(display)
            renderer
        } catch (_: Throwable) {
            "ARM64 Mobile GPU"
        }
    }
}
