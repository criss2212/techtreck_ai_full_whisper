import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:record/record.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

typedef WhisperEventHandler = void Function(Map<String, dynamic> event);

class WhisperWsTranscriber {
  final Uri wsUri;
  final WhisperEventHandler onEvent;
  WebSocketChannel? _ch;
  final _rec = AudioRecorder();
  StreamSubscription<Uint8List>? _streamSub;

  WhisperWsTranscriber(this.wsUri, {required this.onEvent});

  Future<bool> checkPermission() async {
    return await _rec.hasPermission();
  }

  Future<void> start() async {
    _ch = WebSocketChannel.connect(wsUri);
    // Start PCM stream at 16k mono
    final stream = await _rec.startStream(RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: 16000,
      numChannels: 1,
      bitRate: 256000,
    ));
    _streamSub = stream.listen((data) {
      // send as base64 chunk
      _ch?.sink.add(jsonEncode({
        "type": "audio_chunk",
        "format": kIsWeb ? "webm_opus" : "pcm_s16le",
        "sample_rate": 16000,
        "content": base64Encode(data),
      }));
    });
    // listen events
    _ch?.stream.listen((msg) {
      try {
        final evt = jsonDecode(msg as String) as Map<String, dynamic>;
        onEvent(evt);
      } catch (_) {}
    });
  }

  Future<void> stop() async {
    try {
      await _rec.stop();
    } catch (_) {}
    await _streamSub?.cancel();
    _streamSub = null;
    await _ch?.sink.close();
    _ch = null;
  }

  Future<void> dispose() async {
    await stop();
  }
}
