package com.aplikasi.deteksihoax.ui.home

import android.graphics.drawable.Drawable
import android.os.Bundle
import android.util.Log
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.databinding.BindingAdapter
import androidx.databinding.DataBindingUtil
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.swiperefreshlayout.widget.CircularProgressDrawable
import com.aplikasi.deteksihoax.MainActivity
import com.aplikasi.deteksihoax.R
import com.aplikasi.deteksihoax.databinding.FragmentHomeBinding
import com.aplikasi.deteksihoax.network.ApiClient
import com.aplikasi.deteksihoax.utils.DetectionModel
import com.aplikasi.deteksihoax.utils.HistoryDetectionModel
import com.aplikasi.deteksihoax.utils.PredictModel
import com.aplikasi.deteksihoax.utils.RecomModel
import com.google.android.material.button.MaterialButton
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase

class HomeFragment : Fragment(), HomeView {

    private lateinit var binding: FragmentHomeBinding
    private lateinit var presenter: HomePresenter
    private lateinit var recomAdapter: RecomAdapter
    val db = Firebase.firestore
    var teksBerita = ""
    var judulBerita = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = FragmentHomeBinding.inflate(layoutInflater)
        // Inflate the layout for this fragment
        presenter = HomePresenter(this, ApiClient.apiService)

        binding.btnDeteksi.setOnClickListener {
            val title = binding.etJudul.text.toString()
            val newsText = binding.etBerita.text.toString()
            if (title.isEmpty() || newsText.isEmpty()) {
                binding.etJudul.setError("Mohon Isi")
                binding.etBerita.setError("Mohon Isi")
                Toast.makeText(requireContext(), "Invalid!", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            judulBerita = title
            teksBerita = newsText

            presenter.sendSot(requireContext(), title, newsText)

        }
        return binding.root


    }

    override fun setupListener() {
        recomAdapter = RecomAdapter()
        recomAdapter.context = requireActivity()
        binding.rvRecomendations.adapter = recomAdapter
        binding.rvRecomendations.layoutManager = LinearLayoutManager(requireActivity())
    }

    override fun homeLoading(loading: Boolean) {
        if(loading) binding.llResult.visibility = View.GONE
        binding.btnDeteksi.setShowProgress(loading)
        binding.btnDeteksi.isEnabled = !loading
    }

    override fun sendPredict(response: PredictModel) {
        Log.e("sendPredict", "" + response)
    }

    override fun predictMessage(message: String) {

        Log.e("predictMessage", "" + message)
    }

    override fun sendRecommendation(response: PredictModel) {
        Log.e("sendRecommendation", "" + response)
//        val status = response.results?.get(0)
//        val persentase = response.results?.get(1)
//        val content = "Berita yang anda masukkan" + persentase + " terindikasi hoax,\n" +
//                "berikut beberapa rekomendasi bacaan yang dapat\n" +
//                "mempertimbangkan berita tersebut."

        try {
            binding.tvHasil.text = response.predicition_result
            binding.llResult.visibility = View.VISIBLE

            recomAdapter.updateData(response.recomendations!!)

            saveContent(judulBerita, teksBerita, response.predicition_result!!, response.recomendations!!)
        }catch (e: java.lang.Exception){
            Toast.makeText(requireContext(), "" + e.message, Toast.LENGTH_SHORT).show()

        }


    }

    @BindingAdapter("showProgress")
    fun MaterialButton.setShowProgress(showProgress: Boolean?) {
        icon = if (showProgress == true) {
            CircularProgressDrawable(requireContext()).apply {
                setStyle(CircularProgressDrawable.DEFAULT)
                setColorSchemeColors(ContextCompat.getColor(requireContext(), R.color.white))
                start()
            }
        } else null
        if (icon != null) { // callback to redraw button icon
            icon.callback = object : Drawable.Callback {
                override fun unscheduleDrawable(who: Drawable, what: Runnable) {
                }

                override fun invalidateDrawable(who: Drawable) {
                    this@setShowProgress.invalidate()
                }

                override fun scheduleDrawable(who: Drawable, what: Runnable, `when`: Long) {
                }
            }
        }
    }

    fun saveContent(title: String, content: String, detectionResult:String, recommendation: ArrayList<String>){
        // Add a new document with a generated ID
        val detectionRef = db.collection("history_detection")

        val detectionData = HistoryDetectionModel(

            detectionResult,
            content,
            title,
            recommendation
            )

        detectionRef.add(detectionData)
            .addOnSuccessListener {
                Toast.makeText(requireActivity(), "DocumentSnapshot added with ID: ${it.id}" , Toast.LENGTH_SHORT).show()
                Log.e("SaveDetection", "DocumentSnapshot added with ID: ${it.id}")
            }
            .addOnFailureListener {
                Toast.makeText(requireActivity(), "DocumentSnapshot added with ID: ${it.message}" , Toast.LENGTH_SHORT).show()
                Log.e("SaveDetection", "Error adding document", it)
            }
    }
}