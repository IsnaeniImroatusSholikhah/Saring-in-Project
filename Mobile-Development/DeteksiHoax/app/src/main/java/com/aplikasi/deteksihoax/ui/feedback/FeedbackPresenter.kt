package com.aplikasi.deteksihoax.ui.feedback

import android.content.Context
import android.util.Log
import android.widget.Toast
import com.aplikasi.deteksihoax.network.ApiService
import com.aplikasi.deteksihoax.ui.home.HomeView
import com.aplikasi.deteksihoax.utils.DetectionModel
import com.aplikasi.deteksihoax.utils.FeedbackModel
import com.aplikasi.deteksihoax.utils.PredictModel
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class FeedbackPresenter(
    private val view: FeedbackView,
    private val api: ApiService,
    ) {

    init {
        view.setupListener()
    }

    fun sendFeedback(rating: String, message: String){
        val db = Firebase.firestore
        view.feedbackLoading(true)
        try{
            GlobalScope.launch {
                val feedbackRef = db.collection("feedback")

                val feedbackData = FeedbackModel(
                    message,
                    rating
                )

                feedbackRef.add(feedbackData)
                    .addOnSuccessListener {
                        view.feedbackMessage("Berhasil Tambah Feedback!")
                    }
                    .addOnFailureListener {
                        view.feedbackMessage("Gagal Tambah Feedback!")
                    }
                    .addOnCompleteListener {
                        view.feedbackLoading(false)
                    }
            }
        }catch (e: Exception){
            view.feedbackMessage(e.message!!)
        }
    }
}

interface FeedbackView {
    fun setupListener()
    fun feedbackLoading(loading: Boolean)
    fun sendFeedback(response: PredictModel)
    fun feedbackMessage(message: String)
}