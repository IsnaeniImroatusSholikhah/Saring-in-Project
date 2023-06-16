package com.aplikasi.deteksihoax.ui.home

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.databinding.DataBindingUtil
import androidx.recyclerview.widget.RecyclerView
import com.aplikasi.deteksihoax.databinding.ItemRecomBinding

class RecomAdapter(): RecyclerView.Adapter<RecomAdapter.RecomViewHolder>() {

    private val recomList = ArrayList<String>()
    var context: Context? = null

    class RecomViewHolder(
        val binding: ItemRecomBinding,
    ): RecyclerView.ViewHolder(binding.root){
        fun bind(recom: String, context: Context?){

            binding.tvRekomendasi.text = recom

            binding.tvRekomendasi.setOnClickListener {
                val link = "${recom}"
                val i = Intent(Intent.ACTION_VIEW)
                i.data = Uri.parse(link)
                context?.startActivity(i)
            }

        }
    }

    override fun onCreateViewHolder(
        parent: ViewGroup,
        viewType: Int
    ): RecomAdapter.RecomViewHolder {
        val layoutInflater = LayoutInflater.from(parent.context)
        val binding: ItemRecomBinding = ItemRecomBinding.inflate(layoutInflater, parent, false)
        return RecomAdapter.RecomViewHolder(binding)
    }

    override fun onBindViewHolder(holder: RecomAdapter.RecomViewHolder, position: Int) {
        holder.bind(
            recomList[position],
            context
        )
    }

    override fun getItemCount(): Int {
        return recomList.size
    }

    fun updateData(recom: ArrayList<String>) {
        recomList.clear()
        recomList.addAll(recom)
        notifyDataSetChanged()
    }
}