package com.aplikasi.deteksihoax.ui.history

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import com.aplikasi.deteksihoax.R
import com.aplikasi.deteksihoax.databinding.FragmentHistoryBinding
import com.aplikasi.deteksihoax.databinding.FragmentHomeBinding
import com.aplikasi.deteksihoax.network.ApiClient
import com.aplikasi.deteksihoax.ui.home.HomePresenter
import com.aplikasi.deteksihoax.ui.home.RecomAdapter
import com.aplikasi.deteksihoax.utils.DetectionModel
import com.aplikasi.deteksihoax.utils.HistoryDetectionModel
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase

class HistoryFragment : Fragment(), HistoryView {

    private lateinit var binding: FragmentHistoryBinding
    private lateinit var presenter: HistoryPresenter
    private lateinit var historyAdapter: HistoryAdapter
    val db = Firebase.firestore

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = FragmentHistoryBinding.inflate(layoutInflater)
        // Inflate the layout for this fragment
        presenter = HistoryPresenter(this, ApiClient.apiService)

        historyAdapter = HistoryAdapter()
        binding.rvHistory.adapter = historyAdapter
        historyAdapter.context = requireActivity()
        binding.rvHistory.layoutManager = LinearLayoutManager(requireActivity())
        presenter.fetchHistory(requireContext())

        binding.srlRefresh.setOnRefreshListener {
            presenter.fetchHistory(requireContext())
        }

        return binding.root
    }

    override fun setupListener() {
    }

    override fun historyLoading(loading: Boolean) {
        binding.srlRefresh.isRefreshing = loading
    }

    override fun fetchHistory(response: ArrayList<HistoryDetectionModel>) {
        historyAdapter.updateData(response)
    }

    override fun historyMessage(message: String) {
        Toast.makeText(requireActivity(), "" + message, Toast.LENGTH_SHORT).show()
    }
}