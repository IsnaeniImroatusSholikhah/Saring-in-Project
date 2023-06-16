package com.aplikasi.deteksihoax.ui.home

import android.content.Context
import android.util.Log
import com.aplikasi.deteksihoax.network.ApiService
import com.aplikasi.deteksihoax.utils.PredictModel
import com.aplikasi.deteksihoax.utils.RecomModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class HomePresenter(
    private val view: HomeView,
    private val api: ApiService,
) {

    init {
        view.setupListener()
    }

    fun sendSot(context: Context, title: String, newsText: String){
        view.homeLoading(true)
        try{
            GlobalScope.launch {
                val response = api.getSot(title)
                if (response.isSuccessful) {
                    withContext(Dispatchers.Main) {
                        val data = response.body()
                        if (data != null) {
                            if(data.news_source_of_truth?.size!! > 0){
                                 sendPredict(data.news_source_of_truth?.get(0) ?: "", newsText)
                        } else {
                            sendPredict("", newsText)
                        }
                    }
                        view.homeLoading(false)
                        view.predictMessage("Pengambilan Sumber Terpercaya Sudah Berhasil!")
                    }
                } else {
                    view.predictMessage("Terjadi kesalahan")
                }
            }
        }catch (e: Exception){
            view.predictMessage(e.message!!)
        }
    }

    fun sendPredict(sourceOfTruth: String, newsText: String){
        view.homeLoading(true)
        try{
            GlobalScope.launch {
                val response = api.predict(sourceOfTruth, newsText)
                if (response.isSuccessful) {
                    withContext(Dispatchers.Main) {
                        val data = response.body()
                        if(data != null) {
                            val prediction_result = data.predicition_result
                            if (prediction_result != null) {
                                sendRecommendation(prediction_result, newsText)
                            }
                        }
                        view.homeLoading(false)
                        view.predictMessage("Pengambilan Sumber Terpercaya Sudah Berhasil!")
                    }
                } else {
                    view.predictMessage("Terjadi kesalahan")
                }
            }
        }catch (e: Exception){
            view.predictMessage(e.message!!)
        }
    }

    fun sendRecommendation(prediction_result: String, newsText: String){
        view.homeLoading(true)
        try{
            GlobalScope.launch {
                val response = api.getRecommendation(newsText)
                if (response.isSuccessful) {
                    withContext(Dispatchers.Main) {
                        val data = response.body()
                        if(data != null) {
                            val newPredictModel = PredictModel(prediction_result, data.recomendations, null)
                            view.sendRecommendation(newPredictModel)
                        }
                        view.homeLoading(false)
                        view.predictMessage("Pengambilan Sumber Terpercaya Sudah Berhasil!")
                    }
                } else {
                    view.predictMessage("Terjadi kesalahan")
                }
            }
        }catch (e: Exception){
            view.predictMessage(e.message!!)
        }
    }

}

interface HomeView {
    fun setupListener()
    fun homeLoading(loading: Boolean)
    fun sendPredict(response: PredictModel)
    fun sendRecommendation(response: PredictModel)
    fun predictMessage(message: String)
}