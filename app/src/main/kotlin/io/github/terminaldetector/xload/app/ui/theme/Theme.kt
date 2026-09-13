package io.github.terminaldetector.xload.app.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext

private val LightColors = lightColorScheme(
    primary = XloadPrimary,
    onPrimary = XloadOnPrimary,
    primaryContainer = XloadPrimaryContainer,
    onPrimaryContainer = XloadOnPrimaryContainer,
    secondary = XloadSecondary,
    onSecondary = XloadOnSecondary,
    background = XloadBackground,
    onBackground = XloadOnBackground,
    surface = XloadSurface,
    onSurface = XloadOnSurface,
    error = XloadError,
    onError = XloadOnError,
)

private val DarkColors = darkColorScheme(
    primary = XloadPrimaryDark,
    onPrimary = XloadOnPrimaryDark,
    primaryContainer = XloadPrimaryContainerDark,
    onPrimaryContainer = XloadOnPrimaryContainerDark,
    background = XloadBackgroundDark,
    onBackground = XloadOnBackgroundDark,
    surface = XloadSurfaceDark,
    onSurface = XloadOnSurfaceDark,
)

/**
 * Uses Material You dynamic color on Android 12+ (falls back to the fixed
 * Xload palette below API 31), matching the platform themes most users expect.
 */
@Composable
fun XloadTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit,
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColors
        else -> LightColors
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography(),
        content = content,
    )
}
