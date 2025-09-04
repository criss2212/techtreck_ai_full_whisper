import 'dart:async';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:vibration/vibration.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt; 
import 'ai/ws_transcriber.dart'; 

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const TechTreckApp());
}

class TechTreckApp extends StatelessWidget {
  const TechTreckApp({super.key});
  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => AppState()..loadFromPrefs(),
      child: Consumer<AppState>(
        builder: (context, state, _) {
          final theme = ThemeData(
            brightness: state.highContrast ? Brightness.dark : Brightness.light,
            colorScheme: ColorScheme.fromSeed(
              seedColor: state.accentColor,
              brightness:
                  state.highContrast ? Brightness.dark : Brightness.light,
              //highContrastDark: state.highContrast,
              //highContrastLight: state.highContrast,
            ),
            textTheme: Typography.blackMountainView.apply().copyWith(
                  bodyLarge: TextStyle(fontSize: 16 * state.fontScale),
                  bodyMedium: TextStyle(fontSize: 14 * state.fontScale),
                  bodySmall: TextStyle(fontSize: 12 * state.fontScale),
                ),
            useMaterial3: true,
          );
          return MaterialApp(
            title: 'TechTreck GenAI (Whisper)',
            theme: theme,
            home: const HomeShell(),
            debugShowCheckedModeBanner: false,
          );
        },
      ),
    );
  }
}

class HomeShell extends StatefulWidget {
  const HomeShell({super.key});
  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int idx = 0;
  final pages = const [TranscribeScreen(), HistoryScreen(), SettingsScreen()];
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(child: pages[idx]),
      bottomNavigationBar: NavigationBar(
        selectedIndex: idx,
        onDestinationSelected: (i) => setState(() => idx = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.mic), label: 'Transcriere'),
          NavigationDestination(icon: Icon(Icons.history), label: 'Istoric'),
          NavigationDestination(icon: Icon(Icons.settings), label: 'Setari'),
        ],
      ),
    );
  }
}

class AppState extends ChangeNotifier {
  double fontScale = 1.2;
  bool highContrast = true;
  bool vibrateOnSentence = false;
  bool useAiStt = true;
  Color accentColor = Colors.blue;
  String apiBaseUrl = "http://127.0.0.1:8000";

  final List<TranscriptEntry> history = [];

  Future<void> loadFromPrefs() async {
    final sp = await SharedPreferences.getInstance();
    fontScale = sp.getDouble('fontScale') ?? 1.2;
    highContrast = sp.getBool('highContrast') ?? true;
    vibrateOnSentence = sp.getBool('vibrateOnSentence') ?? false;
    useAiStt = sp.getBool('useAiStt') ?? true;
    apiBaseUrl = sp.getString('apiBaseUrl') ?? "http://127.0.0.1:8000";
    final color = sp.getInt('accentColor');
    if (color != null) accentColor = Color(color);

    final raw = sp.getStringList('history') ?? [];
    history
      ..clear()
      ..addAll(raw.map(TranscriptEntry.fromPersisted));
    notifyListeners();
  }

  Future<void> savePrefs() async {
    final sp = await SharedPreferences.getInstance();
    await sp.setDouble('fontScale', fontScale);
    await sp.setBool('highContrast', highContrast);
    await sp.setBool('vibrateOnSentence', vibrateOnSentence);
    await sp.setBool('useAiStt', useAiStt);
    await sp.setInt('accentColor', accentColor.value);
    await sp.setString('apiBaseUrl', apiBaseUrl);
    await sp.setStringList('history', history.map((e) => e.persist()).toList());
  }

  void updateFontScale(double v) {
    fontScale = v;
    savePrefs();
    notifyListeners();
  }

  void setContrast(bool v) {
    highContrast = v;
    savePrefs();
    notifyListeners();
  }

  void setVibration(bool v) {
    vibrateOnSentence = v;
    savePrefs();
    notifyListeners();
  }

  void setUseAiStt(bool v) {
    useAiStt = v;
    savePrefs();
    notifyListeners();
  }

  void setAccent(Color c) {
    accentColor = c;
    savePrefs();
    notifyListeners();
  }

  void setApiBaseUrl(String s) {
    apiBaseUrl = s;
    savePrefs();
    notifyListeners();
  }

  void addToHistory(TranscriptEntry e) {
    history.insert(0, e);
    savePrefs();
    notifyListeners();
  }

  void deleteFromHistory(TranscriptEntry e) {
    history.remove(e);
    savePrefs();
    notifyListeners();
  }
}

class TranscriptEntry {
  final DateTime ts;
  final String text;
  TranscriptEntry(this.ts, this.text);
  String persist() => '${ts.toIso8601String()}||$text';
  static TranscriptEntry fromPersisted(String s) {
    final i = s.indexOf('||');
    return TranscriptEntry(
        DateTime.parse(s.substring(0, i)), s.substring(i + 2));
  }
}

class TranscribeScreen extends StatefulWidget {
  const TranscribeScreen({super.key});
  @override
  State<TranscribeScreen> createState() => _TranscribeScreenState();
}

class _TranscribeScreenState extends State<TranscribeScreen> {
  final stt.SpeechToText _stt = stt.SpeechToText(); // fallback
  bool _available = false;
  bool _listening = false;
  String _partial = '';
  final List<String> _sentences = [];
  Timer? _flushTimer;
  WhisperWsTranscriber? _ws;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    final useAi = context.read<AppState>().useAiStt;

    if (useAi) {
      final base = context.read<AppState>().apiBaseUrl;

      final uri = Uri.parse(base);
      final wsUri = uri.replace(
        scheme: uri.scheme == 'https' ? 'wss' : 'ws',
        path: '/ws',
        query: null,
      );

      _ws = WhisperWsTranscriber(wsUri, onEvent: (evt) {
        final type = (evt['type'] as String?) ?? '';
        final text = (evt['text'] as String?) ?? '';
        if (text.isEmpty) return;
        _ingest(text, isFinal: type != 'partial');
      });

      // pe Web permisiunea e cerută de browser; hasPermission poate întoarce false
      _available = kIsWeb ? true : (await _ws!.checkPermission());
    } else {
      _available = await _stt.initialize(
        onStatus: (s) {
          if (s.contains('notListening') && _listening) _start();
        },
        onError: (e) => debugPrint('STT error: $e'),
      );
    }
    setState(() {});
  }

  void _start() async {
    if (!_available) return;
    _listening = true;
    setState(() {});
    if (context.read<AppState>().useAiStt) {
      await _ws?.start();
    } else {
      await _stt.listen(
        localeId: await _pickRoLocale(),
        partialResults: true,
        listenMode: stt.ListenMode.dictation,
        onResult: (res) =>
            _ingest(res.recognizedWords, isFinal: res.finalResult),
      );
    }
  }

  Future<String?> _pickRoLocale() async {
    try {
      final locales = await _stt.locales();
      final ro = locales.firstWhere(
        (l) => l.localeId.toLowerCase().startsWith('ro'),
        orElse: () => locales.first,
      );
      return ro.localeId;
    } catch (_) {
      return null;
    }
  }

  void _stop() async {
    _listening = false;
    setState(() {});
    if (context.read<AppState>().useAiStt) {
      await _ws?.stop();
    } else {
      await _stt.stop();
    }
  }

  void _ingest(String text, {required bool isFinal}) {
    final rx = RegExp(r"([^.!?\n]+[.!?])");
    final matches = rx.allMatches(text).toList();
    if (matches.isNotEmpty) {
      for (final m in matches) {
        final sentence = m.group(0)!.trim();
        if (sentence.isNotEmpty) _pushSentence(sentence);
      }
      _partial = text.substring(matches.last.end).trim();
    } else {
      _partial = text.trim();
    }

    _flushTimer?.cancel();
    _flushTimer = Timer(const Duration(milliseconds: 1200), () {
      if (_partial.isNotEmpty) {
        _pushSentence(_partial);
        _partial = '';
        setState(() {});
      }
    });

    if (isFinal && _partial.isNotEmpty) {
      _pushSentence(_partial);
      _partial = '';
    }

    setState(() {});
  }

  Future<void> _pushSentence(String s) async {
    _sentences.add(s);
    final state = context.read<AppState>();
    if (state.vibrateOnSentence &&
        !kIsWeb &&
        (await Vibration.hasVibrator() ?? false)) {
      Vibration.vibrate(duration: 30, amplitude: 128);
    }
  }

  void _clear() {
    _sentences.clear();
    _partial = '';
    setState(() {});
  }

  Future<void> _saveSession() async {
    final text =
        [..._sentences, if (_partial.isNotEmpty) _partial].join(' ').trim();
    if (text.isEmpty) return;
    context
        .read<AppState>()
        .addToHistory(TranscriptEntry(DateTime.now(), text));
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Sesiune salvata in istoric')));
    }
  }

  @override
  void dispose() {
    _flushTimer?.cancel();
    _stt.stop();
    _ws?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final canListen = _available && !_listening;
    final fullText =
        ([..._sentences, if (_partial.isNotEmpty) _partial]).join('\n');
    final useAi = context.watch<AppState>().useAiStt;

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            spacing: 12,
            runSpacing: 8,
            children: [
              ElevatedButton.icon(
                onPressed: canListen ? _start : null,
                icon: const Icon(Icons.mic),
                label:
                    Text(useAi ? 'Porneste (Whisper AI)' : 'Porneste (OS STT)'),
              ),
              ElevatedButton.icon(
                onPressed: _listening ? _stop : null,
                icon: const Icon(Icons.stop),
                label: const Text('Opreste'),
              ),
              OutlinedButton.icon(
                onPressed: _saveSession,
                icon: const Icon(Icons.save_alt),
                label: const Text('Salveaza'),
              ),
              TextButton.icon(
                onPressed: _clear,
                icon: const Icon(Icons.clear_all),
                label: const Text('Curata'),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Expanded(
            child: SingleChildScrollView(
              child: SelectableText(fullText.isEmpty ? ' ' : fullText),
            ),
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(_available
                  ? (_listening ? 'Ascult…' : 'Pregatit')
                  : 'Serviciu indisponibil'),
              Text(useAi
                  ? 'Mod: Whisper AI'
                  : (kIsWeb ? 'Mod: Web OS STT' : 'Mod: OS STT')),
            ],
          )
        ],
      ),
    );
  }
}

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});
  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.all(16),
          child: TextField(
            decoration: const InputDecoration(
                prefixIcon: Icon(Icons.search), hintText: 'Cauta in istoric…'),
            onChanged: (v) =>
                state.notifyListeners(), // trigger rebuild below via watch
          ),
        ),
        Expanded(
          child: state.history.isEmpty
              ? const Center(child: Text('Nimic in istoric inca'))
              : ListView.builder(
                  itemCount: state.history.length,
                  itemBuilder: (c, i) {
                    final e = state.history[i];
                    return Dismissible(
                      key: ValueKey(e.persist()),
                      background: Container(color: Colors.redAccent),
                      onDismissed: (_) => state.deleteFromHistory(e),
                      child: ListTile(
                        title: Text(_preview(e.text)),
                        subtitle: Text(_fmt(e.ts)),
                      ),
                    );
                  },
                ),
        ),
      ],
    );
  }

  String _preview(String s) => s.length > 80 ? '${s.substring(0, 80)}…' : s;
  String _fmt(DateTime d) {
    String two(int x) => x.toString().padLeft(2, '0');
    return '${two(d.day)}.${two(d.month)}.${d.year}  ${two(d.hour)}:${two(d.minute)}';
  }
}

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});
  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final urlCtrl = TextEditingController(text: state.apiBaseUrl);
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        const Text('Accesibilitate',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        Row(
          children: [
            const Text('Marime font'),
            Expanded(
              child: Slider(
                value: state.fontScale,
                min: 0.8,
                max: 2.0,
                divisions: 12,
                label: state.fontScale.toStringAsFixed(1),
                onChanged: (v) => context.read<AppState>().updateFontScale(v),
              ),
            ),
          ],
        ),
        SwitchListTile(
          value: state.highContrast,
          onChanged: (v) => context.read<AppState>().setContrast(v),
          title: const Text('Contrast ridicat'),
          subtitle: const Text('Fundal intunecat, text clar'),
        ),
        SwitchListTile(
          value: state.vibrateOnSentence,
          onChanged: (v) => context.read<AppState>().setVibration(v),
          title: const Text('Vibratie la propozitie (non-web)'),
        ),
        const Divider(),
        const Text('AI / STT',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        SwitchListTile(
          value: state.useAiStt,
          onChanged: (v) => context.read<AppState>().setUseAiStt(v),
          title: const Text('Foloseste Whisper AI STT'),
          subtitle: const Text('Necesita serverul AI pornit la API Base URL'),
        ),
        TextField(
          controller: urlCtrl,
          decoration: const InputDecoration(
              prefixIcon: Icon(Icons.link),
              labelText: 'API Base URL (ex: http://127.0.0.1:8000)'),
          onSubmitted: (v) => context.read<AppState>().setApiBaseUrl(v.trim()),
        ),
      ],
    );
  }
}
