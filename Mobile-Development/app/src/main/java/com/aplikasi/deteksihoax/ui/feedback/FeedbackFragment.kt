package com.aplikasi.deteksihoax.ui.feedback

import android.R
import android.content.ContentValues.TAG
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.fragment.app.Fragment
import com.aplikasi.deteksihoax.databinding.FragmentFeedbackBinding
import com.aplikasi.deteksihoax.network.ApiClient
import com.aplikasi.deteksihoax.ui.history.HistoryAdapter
import com.aplikasi.deteksihoax.ui.history.HistoryPresenter
import com.aplikasi.deteksihoax.utils.PredictModel
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase


class FeedbackFragment : Fragment(), FeedbackView {

    private lateinit var binding: FragmentFeedbackBinding
    private lateinit var presenter: FeedbackPresenter
    val db = Firebase.firestore


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = FragmentFeedbackBinding.inflate(layoutInflater)

        presenter = FeedbackPresenter(this, ApiClient.apiService)

        binding.btnSend.setOnClickListener {
            val rating = binding.simpleRatingBar.toString()
            val message = binding.etReview.text.toString()

            presenter.sendFeedback(rating, message)
        }
        return binding.root
    }

    override fun setupListener() {

    }

    override fun feedbackLoading(loading: Boolean) {
        if(!loading) binding.etReview.setText("")
    }

    override fun sendFeedback(response: PredictModel) {

    }

    override fun feedbackMessage(message: String) {
        Toast.makeText(requireContext(), "" + message, Toast.LENGTH_SHORT).show()
    }
}