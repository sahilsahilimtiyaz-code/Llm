package com.localcore.ai

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat

/**
 * Android ForegroundService keeping GGUF downloads and long offline token generations
 * alive while the screen dims, holding a bounded PARTIAL_WAKE_LOCK and displaying progress.
 */
class ModelDownloadService : Service() {

    companion object {
        const val CHANNEL_ID = "localcore_gguf_engine"
        const val NOTIFICATION_ID = 4102
        const val ACTION_UPDATE = "com.localcore.ai.ACTION_UPDATE"
        const val ACTION_STOP = "com.localcore.ai.ACTION_STOP"
        const val EXTRA_TITLE = "extra_title"
        const val EXTRA_DETAIL = "extra_detail"
        const val EXTRA_PROGRESS = "extra_progress"
    }

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        createChannel()
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "LocalCoreAI::GGUFEngineLock").apply {
            setReferenceCounted(false)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                releaseWakeLock()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
                return START_NOT_STICKY
            }
            else -> {
                val title = intent?.getStringExtra(EXTRA_TITLE) ?: "LocalCore AI Engine"
                val detail = intent?.getStringExtra(EXTRA_DETAIL) ?: "Running offline GGUF inference"
                val progress = intent?.getIntExtra(EXTRA_PROGRESS, -1) ?: -1

                if (wakeLock?.isHeld == false) {
                    wakeLock?.acquire(30 * 60 * 1000L) // 30-min safety ceiling
                }
                startForeground(NOTIFICATION_ID, buildNotification(title, detail, progress))
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        releaseWakeLock()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun releaseWakeLock() {
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            val channel = NotificationChannel(
                CHANNEL_ID,
                "LocalCore GGUF Downloads & Inference",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Shows progress for offline .gguf downloads and active GPU generation."
            }
            nm.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(title: String, detail: String, progress: Int): Notification {
        val openIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle(title)
            .setContentText(detail)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(pendingIntent)

        if (progress in 0..100) {
            builder.setProgress(100, progress, false)
        }
        return builder.build()
    }
}
