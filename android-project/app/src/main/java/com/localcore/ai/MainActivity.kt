package com.localcore.ai

import android.annotation.SuppressLint
import android.content.ComponentCallbacks2
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.View
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

/**
 * MainActivity hosting the hardware-accelerated OLED System WebView,
 * AndroidHardwareBridge, Storage Access Framework (.gguf) picker,
 * edge-to-edge keyboard insets, and ComponentCallbacks2.onTrimMemory() LMK protection.
 */
class MainActivity : ComponentActivity(), ComponentCallbacks2 {

    private lateinit var webView: WebView
    private lateinit var hardwareBridge: AndroidHardwareBridge

    private val safGgufLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri ?: return@registerForActivityResult
        try {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        } catch (_: SecurityException) {
            // Transient permission still valid for current session
        }

        var displayName = "imported-model.gguf"
        var sizeBytes = 0L
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val nameIdx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                val sizeIdx = cursor.getColumnIndex(OpenableColumns.SIZE)
                if (nameIdx >= 0) displayName = cursor.getString(nameIdx) ?: displayName
                if (sizeIdx >= 0) sizeBytes = cursor.getLong(sizeIdx)
            }
        }
        hardwareBridge.notifySafFilePicked(uri.toString(), displayName, sizeBytes)
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        webView = WebView(this).apply {
            setBackgroundColor(0xFF090A0F.toInt())
            setLayerType(View.LAYER_TYPE_HARDWARE, null)
            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = false // Strict SQLite3 native persistence; no WebView localStorage
                databaseEnabled = false
                allowFileAccess = true
                allowContentAccess = true
                cacheMode = WebSettings.LOAD_DEFAULT
                mediaPlaybackRequiresUserGesture = false
            }
            webViewClient = WebViewClient()
            webChromeClient = WebChromeClient()
        }

        hardwareBridge = AndroidHardwareBridge(this, webView)
        webView.addJavascriptInterface(hardwareBridge, "AndroidHardwareBridge")

        setContentView(webView)

        // Edge-to-edge insets + smooth mobile keyboard auto-resize
        ViewCompat.setOnApplyWindowInsetsListener(webView) { view, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            val ime = insets.getInsets(WindowInsetsCompat.Type.ime())
            val bottomPadding = maxOf(systemBars.bottom, ime.bottom)
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, bottomPadding)
            insets
        }

        webView.loadUrl("file:///android_asset/webview/index.html")
    }

    fun launchSafGgufPicker() {
        safGgufLauncher.launch(arrayOf("application/octet-stream", "*/*"))
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        hardwareBridge.notifyTrimMemory(level)
        // Also signal local Python FastAPI engine directly to abort generation / unload KV-cache
        thread(start = true) {
            try {
                val url = URL("http://127.0.0.1:8765/api/hardware/trim-memory?level=$level")
                (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 800
                    readTimeout = 800
                    responseCode
                    disconnect()
                }
            } catch (_: Throwable) {
            }
        }
    }

    override fun onDestroy() {
        hardwareBridge.release()
        webView.destroy()
        super.onDestroy()
    }
}
