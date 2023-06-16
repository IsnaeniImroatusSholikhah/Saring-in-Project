package com.aplikasi.deteksihoax.ui.history

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.aplikasi.deteksihoax.databinding.ItemHistoryBinding
import com.aplikasi.deteksihoax.databinding.ItemRecomBinding
import com.aplikasi.deteksihoax.utils.DetectionModel
import com.aplikasi.deteksihoax.utils.HistoryDetectionModel

class HistoryAdapter(): RecyclerView.Adapter<HistoryAdapter.HistoryViewHolder>() {

    private val historyList = ArrayList<HistoryDetectionModel>()
    var context: Context? = null

    class HistoryViewHolder(
        val binding: ItemHistoryBinding,
    ): RecyclerView.ViewHolder(binding.root){
        fun bind(detectionModel: HistoryDetectionModel, context: Context?){

            binding.tvKonten.text = detectionModel.detection_result

            binding.cvHistory.setOnClickListener {
                val i = Intent(context, DetailHistoryActivity::class.java)
                i.putExtra("berita", detectionModel)
                context?.startActivity(i)
            }

        }
    }

    override fun onCreateViewHolder(
        parent: ViewGroup,
        viewType: Int
    ): HistoryAdapter.HistoryViewHolder {
        val layoutInflater = LayoutInflater.from(parent.context)
        val binding: ItemHistoryBinding = ItemHistoryBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return HistoryAdapter.HistoryViewHolder(binding)
    }

    override fun onBindViewHolder(holder: HistoryAdapter.HistoryViewHolder, position: Int) {
        holder.bind(
            historyList[position],
            context
        )
    }

    override fun getItemCount(): Int {
        return historyList.size
    }

    fun updateData(datas: ArrayList<HistoryDetectionModel>) {
        historyList.clear()
        historyList.addAll(datas)
        notifyDataSetChanged()
    }
}