package com.aplikasi.deteksihoax.ui.history

import android.content.Context
import android.util.Log
import com.aplikasi.deteksihoax.network.ApiService
import com.aplikasi.deteksihoax.ui.home.HomeView
import com.aplikasi.deteksihoax.utils.DetectionModel
import com.aplikasi.deteksihoax.utils.HistoryDetectionModel
import com.aplikasi.deteksihoax.utils.PredictModel
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.firestore.ktx.toObject
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class HistoryPresenter(
    private val view: HistoryView,
    private val api: ApiService,
) {

    init {
        view.setupListener()
    }

    fun fetchHistory(context: Context){
        view.historyLoading(true)
        val db = Firebase.firestore
        val detectionRef = db.collection("history_detection")
        try{
            GlobalScope.launch {
                detectionRef.get()
                    .addOnSuccessListener {
                        val datas = it.toObjects(HistoryDetectionModel::class.java) as ArrayList<HistoryDetectionModel>
                        Log.e("Tes History", "${datas}")
                        view.fetchHistory(datas)
                        view.historyMessage("Berhasil Ambil Data!")
                    }
                    .addOnFailureListener{
                        view.historyMessage(it.message!!)
                    }
                    .addOnCompleteListener {
                        view.historyLoading(false)
                    }
            }
        }catch (e: Exception){
            view.historyMessage(e.message!!)
        }

    }

}

interface HistoryView {
    fun setupListener()
    fun historyLoading(loading: Boolean)
    fun fetchHistory(response: ArrayList<HistoryDetectionModel>)
    fun historyMessage(message: String)
}