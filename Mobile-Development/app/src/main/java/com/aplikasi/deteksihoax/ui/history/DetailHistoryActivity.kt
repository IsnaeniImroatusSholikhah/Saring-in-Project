package com.aplikasi.deteksihoax.ui.history

import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.aplikasi.deteksihoax.databinding.ActivityDetailHistoryBinding
import com.aplikasi.deteksihoax.ui.home.RecomAdapter
import com.aplikasi.deteksihoax.utils.DetectionModel
import com.aplikasi.deteksihoax.utils.HistoryDetectionModel


class DetailHistoryActivity : AppCompatActivity() {
    private lateinit var binding: ActivityDetailHistoryBinding
    private lateinit var detectionModel: HistoryDetectionModel
    private lateinit var recomAdapter: RecomAdapter
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityDetailHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = ""


        detectionModel = intent.getSerializableExtra("berita") as HistoryDetectionModel

        binding.tvBerita.text = detectionModel.news_content
        binding.tvHasil.text = detectionModel.detection_result
        recomAdapter = RecomAdapter()
        recomAdapter.context = this
        binding.rvRecomendations.layoutManager = LinearLayoutManager(this)
        binding.rvRecomendations.adapter = recomAdapter
        detectionModel.recommendation?.let { recomAdapter.updateData(it) }

        val text = "Berita Terkait `${detectionModel.news_title}` menghasilkan ${detectionModel.detection_result}"

        binding.btnShare.setOnClickListener {
            onClickWhatsApp(text)
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressed()
        return true
    }

    fun onClickWhatsApp(message: String) {
        val pm = packageManager
        try {
            val i = Intent(
                Intent.ACTION_SEND
            )
            i.type = "text/plain"
            i.setPackage("com.whatsapp") // so that only Whatsapp reacts and not the chooser

            i.putExtra(Intent.EXTRA_TEXT, message)
            startActivity(i)
        } catch (e: PackageManager.NameNotFoundException) {
            Toast.makeText(this, "WhatsApp not Installed", Toast.LENGTH_SHORT)
                .show()
        }
    }
}