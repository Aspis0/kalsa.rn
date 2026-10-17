package com.rnllama;

import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.system.ErrnoException;
import android.system.Os;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Debug-only passthrough for two ggml Hexagon switches, read from
 * {filesDir}/hexagon-env (KEY=VALUE lines):
 *   GGML_HEXAGON_FA_SELECT  digits only, e.g. 1 = no HMX flash attention
 *   GGML_HEXAGON_OPFILTER   [A-Za-z0-9_|]+, e.g. flash_attn_ext or set_rows
 * Any other key or value is skipped. Only applies to debuggable builds. The file
 * is capped at 4 KB and read once per process, at the first loadNative, so edits
 * need an app restart. A stale file changes NPU behaviour: delete it after experiments.
 */
final class HexagonDebugEnv {
  private static final String TAG = "RNLlama";
  private static final String FILE_NAME = "hexagon-env";
  private static final int MAX_BYTES = 4 * 1024;
  private static final String FA_SELECT = "GGML_HEXAGON_FA_SELECT";
  private static final String OPFILTER = "GGML_HEXAGON_OPFILTER";

  private HexagonDebugEnv() {}

  static void applyIfDebuggable(Context context) {
    if ((context.getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) == 0) return;

    File file = new File(context.getFilesDir(), FILE_NAME);
    if (!file.isFile()) return;

    try {
      byte[] bytes = readBounded(file);
      if (bytes.length > MAX_BYTES) {
        Log.w(TAG, "hexagon-env: ignored, larger than " + MAX_BYTES + " bytes");
        return;
      }
      applyPairs(parse(new String(bytes, StandardCharsets.UTF_8)));
    } catch (Exception e) {
      Log.w(TAG, "hexagon-env: not applied", e);
    }
  }

  private static void applyPairs(Map<String, String> pairs) throws ErrnoException {
    int applied = 0;
    List<String> skipped = new ArrayList<>();
    for (Map.Entry<String, String> entry : pairs.entrySet()) {
      if (!accepts(entry.getKey(), entry.getValue())) {
        skipped.add(entry.getKey());
        continue;
      }
      Os.setenv(entry.getKey(), entry.getValue(), true);
      Log.i(TAG, "hexagon-env: " + entry.getKey() + "=" + entry.getValue());
      applied++;
    }
    Log.i(TAG, "hexagon-env: applied " + applied + ", skipped " + skipped.size()
        + (skipped.isEmpty() ? "" : " " + skipped));
  }

  static Map<String, String> parse(String text) {
    Map<String, String> pairs = new LinkedHashMap<>();
    for (String rawLine : text.split("\n")) {
      String line = rawLine.trim();
      if (line.isEmpty() || line.startsWith("#")) continue;

      int eq = line.indexOf('=');
      if (eq <= 0) continue;

      pairs.put(line.substring(0, eq).trim(), line.substring(eq + 1).trim());
    }
    return pairs;
  }

  static boolean accepts(String key, String value) {
    if (key.equals(FA_SELECT)) return value.matches("[0-9]+");
    if (key.equals(OPFILTER)) return value.matches("[A-Za-z0-9_|]+");
    return false;
  }

  // Reads at most MAX_BYTES + 1 bytes, so the caller can tell "too large" from "exactly at the cap".
  private static byte[] readBounded(File file) throws IOException {
    byte[] buffer = new byte[MAX_BYTES + 1];
    int total = 0;
    try (InputStream in = new FileInputStream(file)) {
      while (total < buffer.length) {
        int n = in.read(buffer, total, buffer.length - total);
        if (n == -1) break;
        total += n;
      }
    }
    return Arrays.copyOf(buffer, total);
  }
}
