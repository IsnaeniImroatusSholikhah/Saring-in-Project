package com.aplikasi.deteksihoax

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import androidx.databinding.DataBindingUtil
import androidx.navigation.NavController
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import com.aplikasi.deteksihoax.databinding.ActivityMainBinding

lateinit var binding: ActivityMainBinding
private lateinit var navController: NavController

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = DataBindingUtil.setContentView(this, R.layout.activity_main)
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(false)

        val navHostFragment = supportFragmentManager.findFragmentById(R.id.home_nav_host) as NavHostFragment
        navController = navHostFragment.navController
        binding.bottoNavigation.setupWithNavController(navController)
        setupActionBarWithNavController(navController)

        NavController.OnDestinationChangedListener { controller, destination, arguments ->
            title = when (destination.id) {
                R.id.homeFragment -> "Home"
                R.id.historyFragment -> "History"
                R.id.feedbackFragment -> "Feedback"
                else -> "Default title"
            }
            supportActionBar?.title = title
        }
    }
    override fun onSupportNavigateUp() = navController.navigateUp()
}