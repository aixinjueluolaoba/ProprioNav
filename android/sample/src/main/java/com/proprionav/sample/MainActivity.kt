package com.proprionav.sample

import android.app.Activity
import android.os.Bundle
import android.widget.TextView
import com.proprionav.ProprioNav

class MainActivity : Activity() {
    private var navigator: ProprioNav? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val output = runCatching {
            val nav = ProprioNav.fromAssets(this)
            navigator = nav
            nav.step(
                posX = 0F,
                posY = 0F,
                targetX = 900F,
                targetY = 500F,
                heading = 0F,
            )
        }
        val text = output.fold(
            onSuccess = {
                "direction=${it.direction}\nspeed=${it.speed}\njump=${it.jump}"
            },
            onFailure = { "ProprioNav initialization failed:\n${it.message}" },
        )
        setContentView(TextView(this).apply {
            textSize = 20F
            setPadding(48, 80, 48, 48)
            this.text = text
        })
    }

    override fun onDestroy() {
        navigator?.close()
        navigator = null
        super.onDestroy()
    }
}
