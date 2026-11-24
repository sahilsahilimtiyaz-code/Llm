# Preserve @JavascriptInterface methods for WebView native bridge
-keepattributes JavascriptInterface
-keepclassmembers class com.localcore.ai.AndroidHardwareBridge {
    @android.webkit.JavascriptInterface <methods>;
}
-keep class com.localcore.ai.ModelDownloadService { *; }
