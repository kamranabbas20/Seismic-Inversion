"""Bilingual content for the post-stack inversion user guide.

Each entry is (kind, english, russian).  Kept apart from the layout code so the
wording can be revised without touching the PDF machinery.

Russian terminology was reviewed term by term with the project owner; the
decisions are recorded in TERMS below so a later edit does not quietly
reintroduce a rejected word.
"""

# Terminology settled with the project owner.  Where English uses two words for
# one idea, Russian deliberately uses one -- those are marked.
TERMS = {
    "post-stack": "суммарные данные",
    "wavelet": "сейсмический импульс",           # never «вейвлет»
    "well tie": "привязка",                      # never «увязка»
    "low-frequency model": "низкочастотная модель",   # never abbreviated
    "background model": "низкочастотная модель",      # same term in Russian
    "blind well": "валидационная скважина",
    "blind validation": "валидация на неиспользованных данных",
    "uplift": "улучшение",
    "checkshot": "данные ВСП",
    "cut-off": "предел",
    "crossplot": "график",
    "tops / markers": "отбивки",                 # one term for both
    "inline": "продольный профиль",
    "crossline": "поперечный профиль",
    "misfit / residual": "невязка",              # one term for both
    "Q factor / absorption": "поглощение",       # one term for both
    "drift (time-depth)": "расхождение",
    "drift (frequency)": "смещение",
    "datum": "уровень приведения",
    # Proposed with this release, not yet reviewed term by term with the owner.
    "seismic attribute": "сейсмический атрибут",
    "multi-attribute prediction": "многоатрибутный прогноз",
    "training error": "ошибка на обучающих данных",
    "validation error": "ошибка на валидационной скважине",   # follows «blind well»
    "overfitting": "переобучение",
    "neural network": "нейронная сеть",
    "operator length": "длина оператора",
}

TITLE_EN = "Post-Stack Seismic Inversion"
TITLE_RU = "Инверсия суммарных сейсмических данных"
SUB_EN = "User guide — every step, every setting"
SUB_RU = "Руководство пользователя — все шаги и настройки"

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------

OVERVIEW = [
    ("h1", "What this tool does", "Что делает эта программа"),
    ("p",
     "It turns a post-stack seismic volume into a cube of <b>acoustic impedance</b> — "
     "a rock property — calibrated against well logs. Impedance sections can be read "
     "as geology in a way amplitude sections cannot: a bright amplitude may be tuning, "
     "but an impedance contrast is a change in the rock.",
     "Программа преобразует куб суммарных сейсмических данных в куб "
     "<b>акустического импеданса</b> — физического свойства породы, калиброванного "
     "по данным ГИС. Разрезы импеданса читаются как геология, в отличие от разрезов "
     "амплитуд: яркая амплитуда может быть эффектом настройки, а контраст импеданса — "
     "это реальное изменение породы."),
    ("p",
     "Four inversion engines are included, plus the QC needed to trust the result: "
     "well-tie optimisation, wavelet estimation, a low-frequency model, blind-well "
     "cross-validation, and uncertainty. Impedance can then be turned into a curve a "
     "well measures — through a calibrated transform, or by predicting it directly "
     "from seismic attributes. The workflow is eleven steps and is meant to be "
     "followed in order — each step consumes what the previous one produced.",
     "Реализованы четыре алгоритма инверсии и весь контроль качества, необходимый, "
     "чтобы результату можно было доверять: оптимизация привязки скважин, оценка "
     "сейсмического импульса, низкочастотная модель, кросс-валидация на "
     "неиспользованных данных и оценка неопределённости. Затем импеданс можно "
     "преобразовать в кривую, измеряемую в скважине, — через калиброванную "
     "зависимость либо прогнозируя её напрямую по сейсмическим атрибутам. Рабочий "
     "процесс состоит из одиннадцати шагов и выполняется по порядку: каждый шаг "
     "использует результат предыдущего."),

    ("h2", "What you need before you start", "Что нужно подготовить заранее"),
    ("bullets",
     ["A <b>post-stack SEG-Y</b> 3D volume (or a 2D line). Up to 1 GB through the browser.",
      "At least one <b>well</b> with a sonic (DT) and a density (RHOB) log, as LAS.",
      "The wells' <b>X/Y coordinates</b> — from the LAS header or a header CSV. "
      "Without them a well cannot be placed on the seismic and is ignored.",
      "Optional but valuable: a <b>checkshot</b> (time-depth), a <b>deviation survey</b>, "
      "formation <b>tops</b>, and interpreted <b>horizons</b> as CSV.",
      "<b>Two or more wells</b> if you want blind validation — with one well no score "
      "can be blind."],
     ["Куб <b>суммарных данных SEG-Y</b> 3D (или 2D-линия). До 1 ГБ через браузер.",
      "Хотя бы одна <b>скважина</b> с акустическим (DT) и плотностным (RHOB) каротажем "
      "в формате LAS.",
      "<b>Координаты X/Y</b> скважин — из заголовка LAS или из CSV-файла заголовков. "
      "Без них скважину нельзя привязать к сейсмике, и она будет проигнорирована.",
      "Желательно: <b>данные ВСП</b> (связь время–глубина), <b>инклинометрия</b>, "
      "<b>отбивки</b> пластов и интерпретированные <b>горизонты</b> в формате CSV.",
      "<b>Две скважины и более</b> — если нужна валидация на неиспользованных данных: "
      "по одной скважине такую оценку получить невозможно."]),

    ("h2", "Starting the application", "Запуск программы"),
    ("code",
     "python -m venv .venv\n"
     ".venv\\Scripts\\activate          # Windows\n"
     "source .venv/bin/activate        # macOS / Linux\n"
     "pip install -r requirements.txt\n"
     "streamlit run app.py",
     "python -m venv .venv\n"
     ".venv\\Scripts\\activate          # Windows\n"
     "source .venv/bin/activate        # macOS / Linux\n"
     "pip install -r requirements.txt\n"
     "streamlit run app.py"),
    ("note",
     "If PowerShell blocks activation with “running scripts is disabled”, run "
     "<b>Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass</b> in that window "
     "first. It applies only to that window and is undone when you close it.",
     "Если PowerShell блокирует активацию сообщением «выполнение сценариев отключено», "
     "выполните сначала <b>Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass</b>. "
     "Команда действует только в текущем окне и отменяется при его закрытии."),
    ("p",
     "The browser opens at <b>localhost:8501</b>. Everything is held in the session, so "
     "moving between steps never reloads the data — but changing something upstream "
     "(the wells, a tie, the volume) deliberately clears everything downstream that "
     "depended on it, rather than leaving a stale result on screen.",
     "Браузер откроется по адресу <b>localhost:8501</b>. Все данные хранятся в сессии, "
     "поэтому переход между шагами не перезагружает их. Однако изменение чего-либо на "
     "раннем шаге (скважины, привязка, куб) намеренно очищает все зависимые результаты, "
     "чтобы на экране не осталось устаревшего результата."),
]

# --------------------------------------------------------------------------
# The eleven steps.  Each: number, title, what it does, what you do, what to
# check, and an optional screenshot with its caption.
# --------------------------------------------------------------------------

STEPS = [
    dict(
        n=1,
        en_title="Data", ru_title="Данные",
        en_does="Loads the seismic volume and the wells. Three sources: a synthetic demo "
                "dataset, a well folder scanned in one pass (F3 layout: Lasfiles / "
                "Checkshot / Track / Tops), or file-by-file upload.",
        ru_does="Загрузка сейсмического куба и скважин. Три источника: синтетический "
                "демонстрационный набор, сканирование папки со скважинами за один проход "
                "(структура F3: Lasfiles / Checkshot / Track / Tops) или загрузка файлов "
                "по одному.",
        en_do="Pick the source. For SEG-Y, check the inline / crossline / X / Y byte "
              "positions against the header scan the app shows. Upload a well-header CSV "
              "if the LAS files carry no coordinates.",
        ru_do="Выберите источник. Для SEG-Y проверьте байтовые позиции продольного "
              "профиля (inline), поперечного профиля (crossline) и координат X / Y по "
              "таблице заголовков, которую показывает программа. Если в LAS нет "
              "координат, загрузите CSV с заголовками скважин.",
        en_check="Trace count, inline and crossline ranges, sample rate and time range are "
                 "plausible. Every well you expect appears, and each has an X/Y.",
        ru_check="Количество трасс, диапазоны продольных и поперечных профилей, шаг "
                 "дискретизации и временной диапазон правдоподобны. Все ожидаемые "
                 "скважины загружены и у каждой есть координаты X/Y.",
        fig="s01_data.png",
        fig_en="Step 1 with the synthetic dataset loaded. The sidebar's session-state table "
               "tracks what exists so far — it is the quickest way to see what a change "
               "upstream has cleared.",
        fig_ru="Шаг 1 с загруженным синтетическим набором. Таблица состояния сессии в "
               "боковой панели показывает, что уже создано, — это самый быстрый способ "
               "увидеть, что было очищено после изменения на раннем шаге.",
    ),
    dict(
        n=2,
        en_title="Seismic viewer", ru_title="Просмотр сейсмики",
        en_does="Inline, crossline, time slice, and an arbitrary traverse through chosen "
                "wells in a chosen order, with gain, clip and colour controls and well "
                "overlays.",
        ru_does="Просмотр продольного профиля (inline), поперечного профиля (crossline), "
                "временного среза и произвольного профиля через выбранные скважины в "
                "заданном порядке, с регулировкой усиления, ограничения и цветовой шкалы, "
                "с отображением скважин.",
        en_do="Look at the data before inverting it. Check polarity, the mute, dead traces "
              "and any acquisition footprint.",
        ru_do="Просмотрите данные до инверсии. Проверьте полярность, зону мьютинга, "
              "«мёртвые» трассы и следы системы наблюдений.",
        en_check="The wells land where you expect on the sections. If a well sits on a dead "
                 "trace or outside the survey, fix it here rather than later.",
        ru_check="Скважины расположены на разрезах там, где ожидается. Если скважина "
                 "попадает на «мёртвую» трассу или вне съёмки, исправьте это сейчас.",
        fig="s02_section.png",
        fig_en="A seismic section with the wells overlaid. Look for polarity, the mute at "
               "the top, and any striping that would follow the acquisition geometry.",
        fig_ru="Сейсмический разрез с нанесёнными скважинами. Оцените полярность, зону "
               "мьютинга в верхней части и полосчатость, повторяющую геометрию системы "
               "наблюдений.",
    ),
    dict(
        n=3,
        en_title="Log QC", ru_title="Контроль качества ГИС",
        en_does="Lists every curve in every LAS with its unit, description, valid "
                "percentage and range. Lets you assign which curve is Vp, which is "
                "density, which is time — and in what unit.",
        ru_does="Список всех кривых во всех LAS с единицами измерения, описанием, долей "
                "валидных значений и диапазоном. Позволяет назначить, какая кривая — Vp, "
                "какая — плотность, какая — время, и в каких единицах.",
        en_do="Check the auto-detected assignment. LAS files are not consistent: mnemonics "
              "vary (DT, DTC, DTCO, AC) and unit strings are often blank or wrong. Correct "
              "anything the app flagged.",
        ru_do="Проверьте автоматическое назначение кривых. LAS-файлы не стандартизованы: "
              "мнемоники различаются (DT, DTC, DTCO, AC), а единицы часто отсутствуют или "
              "указаны неверно. Исправьте всё, что программа отметила.",
        en_check="Vp median 1,500–7,000 m/s and density median 1,800–3,000 kg/m³. A factor "
                 "of ~3.28 means feet and metres are swapped; a factor of 1,000 means g/cm³ "
                 "and kg/m³ are swapped. The depth index unit is also checked — an "
                 "unlabelled index is assumed metric and flagged.",
        ru_check="Медиана Vp — 1 500–7 000 м/с, медиана плотности — 1 800–3 000 кг/м³. "
                 "Расхождение примерно в 3,28 раза означает путаницу футов и метров; в "
                 "1 000 раз — путаницу г/см³ и кг/м³. Единицы глубины также проверяются: "
                 "если они не указаны, принимаются метры с выдачей предупреждения.",
        fig="s03_logqc.png",
        fig_en="Log QC for one well: velocity, density and the impedance they produce, with "
               "the plausible range shaded. A curve leaving the band is almost always a "
               "unit error, not unusual rock.",
        fig_ru="Контроль качества ГИС по одной скважине: скорость, плотность и полученный "
               "из них импеданс; правдоподобный диапазон показан заливкой. Выход кривой за "
               "пределы диапазона почти всегда означает ошибку в единицах измерения, а не "
               "необычную породу.",
    ),
    dict(
        n=4,
        en_title="Well correlation", ru_title="Корреляция скважин",
        en_does="Wells side by side in an order you choose, with logs, formation tops, the "
                "seismic trace at each well, correlation lines between tops, and optional "
                "flattening on a datum.",
        ru_does="Скважины рядом друг с другом в выбранном порядке, с кривыми ГИС, "
                "отбивками, сейсмической трассой у каждой скважины, линиями корреляции "
                "между отбивками и возможностью выравнивания на выбранный уровень "
                "приведения.",
        en_do="Set the well order and pick which curves and tops to show. Flatten on a "
              "marker to compare thickness rather than depth.",
        ru_do="Задайте порядок скважин и выберите отображаемые кривые и отбивки. "
              "Выровняйте разрез на отбивку, чтобы сравнивать толщины, а не глубины.",
        en_check="Tops are consistent from well to well, and the seismic character at each "
                 "well matches the logs. A top that jumps between wells is usually a "
                 "picking error, not geology.",
        ru_check="Отбивки согласованы между скважинами, а волновая картина у каждой "
                 "скважины соответствует кривым ГИС. Резкий скачок отбивки от скважины к "
                 "скважине — обычно ошибка корреляции, а не геология.",
        fig="s04_correlation.png",
        fig_en="Four wells in a chosen order, each with its logs and the seismic trace "
               "extracted at that well, and correlation lines joining the tops.",
        fig_ru="Четыре скважины в заданном порядке; у каждой показаны кривые ГИС и "
               "сейсмическая трасса, извлечённая в точке скважины, а отбивки соединены "
               "линиями корреляции.",
    ),
    dict(
        n=5,
        en_title="Well tie QC", ru_title="Привязка скважин",
        en_does="Compares the well's synthetic seismogram with the trace extracted at the "
                "well, scores the tie, and repairs it: a constant bulk shift, and a "
                "piecewise-linear stretch on top of it.",
        ru_does="Сравнивает синтетическую сейсмограмму по скважине с трассой, извлечённой "
                "в точке скважины, оценивает качество привязки и исправляет её: постоянный "
                "сдвиг и кусочно-линейное растяжение/сжатие поверх него.",
        en_do="Run <b>Optimise tie</b>. A bulk shift fixes a datum error; a stretch fixes a "
              "time-depth that drifts, which is what you get whenever the time-depth comes "
              "from integrating a sonic instead of a checkshot. For deviated wells, switch "
              "on <b>Follow the borehole path</b>.",
        ru_do="Запустите <b>Optimise tie</b> (оптимизация привязки). Постоянный сдвиг "
              "устраняет ошибку уровня приведения; растяжение устраняет накопленное "
              "расхождение зависимости время–глубина, возникающее всегда, когда она "
              "получена интегрированием акустического каротажа, а не по данным ВСП. "
              "Для наклонных скважин включите <b>Follow the borehole path</b> "
              "(по стволу скважины).",
        en_check="Tie correlation. Above ~0.6 is workable; below ~0.4 everything downstream "
                 "will suffer. <b>This step matters more than the choice of inversion "
                 "method</b> — see the validation results at the end of this guide.",
        ru_check="Коэффициент корреляции привязки. Выше ~0,6 — приемлемо; ниже ~0,4 "
                 "пострадают все последующие шаги. <b>Этот шаг важнее выбора алгоритма "
                 "инверсии</b> — см. результаты проверки в конце руководства.",
        fig="s05_tie.png",
        fig_en="The tie panel: well impedance, reflectivity, the extracted seismic against "
               "the synthetic, and the residual. The synthetic should sit on the seismic; "
               "where it does not, the residual shows how badly.",
        fig_ru="Панель привязки: импеданс по скважине, коэффициенты отражения, извлечённая "
               "сейсмическая трасса в сравнении с синтетической и невязка. Синтетическая "
               "трасса должна совпадать с сейсмической; там, где она не совпадает, размер "
               "расхождения виден по невязке.",
        fig2="s05_tietable.png",
        fig2_en="What the optimiser found. The last column is what the stretch was worth "
                "over the bulk shift alone: near zero means the time-depth was not "
                "drifting and the shift was enough — the expected result for a well with "
                "good checkshot data.",
        fig2_ru="Результат работы оптимизатора. В последнем столбце — вклад растяжения по "
                "сравнению с одним лишь постоянным сдвигом: значение около нуля означает, "
                "что расхождения не было и сдвига достаточно; это ожидаемый результат для "
                "скважины с хорошими данными ВСП.",
    ),
    dict(
        n=6,
        en_title="Wavelet", ru_title="Сейсмический импульс",
        en_does="Builds the wavelet: parametric (Ricker, Ormsby, Butterworth), statistical "
                "from the seismic, or least-squares extraction from the wells. Also "
                "estimates bulk Q and can extract a separate wavelet per time window.",
        ru_does="Формирование импульса: параметрический (Рикера, Ормсби, Баттерворта), "
                "статистический по сейсмике или извлечённый по скважинам методом "
                "наименьших квадратов. Также оценивается поглощение (Q) и может "
                "извлекаться отдельный импульс для каждого временного окна.",
        en_do="Prefer <b>well-based extraction</b> — it is the only route that recovers the "
              "true phase of the data. Set the wavelet length to about twice the dominant "
              "period. Check the Q estimate to see whether one wavelet is enough.",
        ru_do="Предпочтительно <b>извлечение по скважинам</b> — только этот способ "
              "восстанавливает истинную фазу данных. Длину импульса задайте примерно в два "
              "раза больше преобладающего периода. Оцените поглощение, чтобы понять, "
              "достаточно ли одного импульса на весь куб.",
        en_check="Amplitude and phase spectra, and the tie correlation the wavelet achieves. "
                 "A frequency drift of a few Hz across the volume is normal; tens of Hz "
                 "means the deep section is being inverted with a wavelet it does not have.",
        ru_check="Амплитудный и фазовый спектры и достигнутая корреляция привязки. "
                 "Смещение частоты на несколько Гц по кубу нормально; десятки Гц означают, "
                 "что глубокая часть разреза инвертируется с неподходящим импульсом.",
        fig="s06_wavelet.png",
        fig_en="The extracted wavelet with its amplitude and phase spectra. Phase is only "
               "drawn where there is enough amplitude to make it meaningful.",
        fig_ru="Извлечённый импульс с амплитудным и фазовым спектрами. Фаза показана "
               "только там, где амплитуда достаточна, чтобы она имела смысл.",
        fig2="s06_spectra.png",
        fig2_en="The wavelet against the seismic and the well reflectivity. The wavelet "
                "should span the seismic's band — much narrower and the inversion will "
                "under-resolve; much wider and it will fit noise.",
        fig2_ru="Импульс в сравнении со спектрами сейсмики и коэффициентов отражения по "
                "скважине. Импульс должен перекрывать полосу частот сейсмики: если он "
                "существенно уже, инверсия недоразрешит разрез, если шире — будет "
                "подстраиваться под шум.",
    ),
    dict(
        n=7,
        en_title="Low-frequency model", ru_title="Низкочастотная модель",
        en_does="Builds the background impedance cube: well impedance low-pass filtered "
                "(default 10 Hz) and interpolated laterally between wells, optionally "
                "guided by horizons.",
        ru_does="Построение фонового куба импеданса: импеданс по скважинам, отфильтрованный "
                "фильтром нижних частот (по умолчанию 10 Гц) и интерполированный между "
                "скважинами, при необходимости с учётом горизонтов.",
        en_do="Load horizons if you have them. <b>One horizon</b> gives a datum shift; "
              "<b>two or more</b> give proportional (layer-cake) flattening, which follows "
              "thickness variation and not just structure.",
        ru_do="Загрузите горизонты, если они есть. <b>Один горизонт</b> задаёт сдвиг к "
              "уровню приведения; <b>два и более</b> дают пропорциональное выравнивание, "
              "учитывающее не только структуру, но и изменение толщин.",
        en_check="The model is smooth, positive everywhere, and matches each well's filtered "
                 "log at that well. Seismic cannot recover frequencies below about 8–10 Hz, "
                 "so this model — not the seismic — carries the absolute impedance level.",
        ru_check="Модель гладкая, всюду положительная и совпадает с отфильтрованной кривой в "
                 "точке каждой скважины. Сейсмика не восстанавливает частоты ниже ~8–10 Гц, "
                 "поэтому именно эта модель, а не сейсмика, задаёт абсолютный уровень "
                 "импеданса.",
        fig="s07_lfm.png",
        fig_en="The low-frequency model as a section. It should be smooth and follow "
               "structure; anything sharp in here came from the interpolation, not from "
               "the rock.",
        fig_ru="Низкочастотная модель в виде разреза. Она должна быть гладкой и следовать "
               "структуре; любые резкие границы здесь порождены интерполяцией, а не "
               "породой.",
    ),
    dict(
        n=8,
        en_title="Inversion", ru_title="Инверсия",
        en_does="Runs one of the four engines over the cube. Offers single-trace QC at a "
                "well, a preview on a small block, and a full-volume run on a background "
                "thread with a progress bar.",
        ru_does="Запуск одного из четырёх алгоритмов по кубу. Доступны контроль по одной "
                "трассе в точке скважины, предварительный расчёт на небольшом блоке и "
                "полный расчёт куба в фоновом потоке с индикатором выполнения.",
        en_do="Always run <b>single-trace QC</b> at a well first, then <b>preview</b> a "
              "block, then the full volume. Use <b>Estimate full-volume runtime</b> before "
              "committing. For sparse-spike, leave the automatic sparsity on.",
        ru_do="Сначала обязательно выполните <b>контроль по одной трассе</b> в точке "
              "скважины, затем <b>предварительный расчёт</b> блока и только потом полный "
              "куб. Перед запуском используйте <b>Estimate full-volume runtime</b> (оценка "
              "времени). Для разреженно-импульсной инверсии оставьте автоподбор "
              "разреженности включённым.",
        en_check="The reported misfit should land near the noise level you specified, not "
                 "near zero. A residual of a fraction of a percent means the inversion is "
                 "fitting noise, not signal.",
        ru_check="Указанная невязка должна быть близка к заданному уровню шума, а не к нулю. "
                 "Невязка в доли процента означает, что инверсия подстраивается под шум, а "
                 "не под сигнал.",
        fig="s08_trace.png",
        fig_en="Single-trace QC — the cheapest check there is. Left: the synthetic against "
               "the recorded trace. Right: inverted impedance against the well log and the "
               "low-frequency model. The inverted curve should add detail to the "
               "background, not depart from it.",
        fig_ru="Контроль по одной трассе — самая быстрая проверка. Слева: синтетическая "
               "трасса в сравнении с записанной. Справа: полученный импеданс в сравнении с "
               "кривой ГИС и низкочастотной моделью. Полученная кривая должна добавлять "
               "детали к фоновой модели, а не уходить от неё.",
    ),
    dict(
        n=9,
        en_title="Blind validation", ru_title="Валидация на неиспользованных данных",
        en_does="Leave-one-out cross-validation. Each well is held out, the low-frequency "
                "model is rebuilt without it, the held-out trace is inverted, and the "
                "result is scored against a log the model has never seen.",
        ru_does="Кросс-валидация с последовательным исключением скважин. Каждая скважина "
                "исключается, низкочастотная модель строится заново без неё, трасса в этой "
                "точке инвертируется, и результат сравнивается с кривой, которую модель "
                "никогда не «видела».",
        en_do="Run it for the engines you are choosing between. It needs at least two "
              "located wells — with one, holding it out leaves nothing to build a "
              "background from.",
        ru_do="Запустите для тех алгоритмов, между которыми выбираете. Требуются минимум две "
              "скважины с координатами: при одной скважине после её исключения не из чего "
              "строить низкочастотную модель.",
        en_check="The <b>uplift</b> column: the correlation the inversion added over its own "
                 "background at a well it had not seen. An engine that cannot beat its own "
                 "background has added nothing, however good the section looks.",
        ru_check="Столбец <b>улучшение</b>: насколько инверсия улучшила результат по "
                 "сравнению с собственной низкочастотной моделью в валидационной скважине. "
                 "Алгоритм, не превосходящий свою низкочастотную модель, не добавил "
                 "ничего, каким бы красивым ни выглядел разрез.",
        fig="s09_verdict.png",
        fig_en="The verdict table. Note the background's own blind score: about 0.00. The "
               "same model scores 0.94 when the well it is being tested against helped "
               "build it — which is exactly why a non-blind number should not be quoted.",
        fig_ru="Итоговая таблица. Обратите внимание на оценку самой низкочастотной модели "
               "на неиспользованных данных: около 0,00. Та же модель даёт 0,94, если "
               "скважина, по которой её проверяют, участвовала в её построении, — именно "
               "поэтому такую оценку приводить нельзя.",
    ),
    dict(
        n=10,
        en_title="Rock property", ru_title="Свойства породы",
        en_does="Turns the result into a curve a well actually measures, by one of two "
                "routes: a transform fitted from log-impedance to a well curve, or "
                "multi-attribute prediction, which goes straight from seismic attributes to "
                "the log and chooses its attributes by the error at a well it has not seen. "
                "Either way the uncertainty is carried through to a cut-off.",
        ru_does="Преобразует результат в кривую, которую действительно измеряют в скважине, "
                "одним из двух путей: через зависимость между логарифмом импеданса и кривой "
                "ГИС либо через многоатрибутный прогноз, который идёт от сейсмических "
                "атрибутов к кривой напрямую и отбирает атрибуты по ошибке на скважине, "
                "которую он не видел. В обоих случаях неопределённость доводится до предела "
                "по свойству.",
        en_do="Pick the route at the top of the page. Transform: choose the curve and the "
              "polynomial degree, fit, predict the cube. Multi-attribute: choose the curve, "
              "the operator length and the attribute limit, fit the predictor, then predict "
              "the cube. Set a cut-off to get a probability map and an expected net thickness.",
        ru_do="Выберите путь в верхней части страницы. Зависимость: выберите кривую и степень "
              "полинома, выполните подбор, рассчитайте куб. Многоатрибутный прогноз: выберите "
              "кривую, длину оператора и предельное число атрибутов, настройте прогноз, затем "
              "рассчитайте куб. Задайте предел, чтобы получить карту вероятности и ожидаемую "
              "эффективную толщину.",
        en_check="For the transform: R² of the fit. Below about 0.3, impedance explains little "
                 "of that curve and the predicted cube will be closer to a constant than to a "
                 "measurement; check the crossplot for one cloud per well, which means the "
                 "relation is well-specific. For multi-attribute: the validation error against "
                 "the spread of the curve — at or above 1.0 the model is no better than "
                 "predicting the mean.",
        ru_check="Для зависимости — коэффициент детерминации R². Ниже ~0,3 импеданс почти не "
                 "объясняет кривую, и расчётный куб будет ближе к константе, чем к измерению; "
                 "на графике проверьте, не образует ли каждая скважина отдельное облако точек "
                 "— это означает, что зависимость индивидуальна для скважины. Для "
                 "многоатрибутного прогноза — отношение ошибки на валидационной скважине к "
                 "разбросу кривой: при значении 1,0 и выше модель не лучше предсказания "
                 "среднего.",
        fig="s10_page.png",
        fig_en="Step 10 with no inversion in the session. The transform route states what it "
               "needs and refuses to guess; the multi-attribute route is available from the "
               "start, because it predicts from the seismic itself rather than from impedance.",
        fig_ru="Шаг 10, когда инверсия ещё не выполнена. Путь через зависимость сообщает, чего "
               "ему не хватает, и не строит догадок; многоатрибутный путь доступен сразу, так "
               "как он прогнозирует по самим сейсмическим данным, а не по импедансу.",
        fig2="s10_multi_fit.png",
        fig2_en="The multi-attribute route after a fit. The summary reports the training and "
                "validation errors side by side, and the metric on the right divides the "
                "validation error by the spread of the curve itself — the single number that "
                "says whether the seismic is carrying this property at all.",
        fig2_ru="Многоатрибутный путь после настройки. В сводке рядом приведены ошибки на "
                "обучающих данных и на валидационной скважине, а показатель справа делит "
                "вторую на разброс самой кривой — это единственное число, отвечающее на "
                "вопрос, несут ли сейсмические данные информацию об этом свойстве.",
    ),
    dict(
        n=11,
        en_title="Results & export", ru_title="Результаты и экспорт",
        en_does="Section viewer, time slices, per-trace QC maps, posterior uncertainty and "
                "P10/P90 volumes where available, crossplots against well logs, and export "
                "to SEG-Y or CSV.",
        ru_does="Просмотр разрезов, временных срезов, карт контроля качества по трассам, "
                "апостериорной неопределённости и кубов P10/P90 (если рассчитаны), "
                "графиков сопоставления с кривыми ГИС и экспорт в SEG-Y или CSV.",
        en_do="Inspect the impedance section, the residual, and the correlation map. Export "
              "the attribute you need; the SEG-Y writer reuses the input file's headers and "
              "coordinate convention.",
        ru_do="Изучите разрез импеданса, невязку и карту корреляции. Экспортируйте нужный "
              "атрибут; при записи SEG-Y используются заголовки и система координат "
              "исходного файла.",
        en_check="The correlation map should be broadly uniform. Systematic low-correlation "
                 "patches usually mean a problem in the data (mute, footprint, dead traces) "
                 "rather than in the inversion.",
        ru_check="Карта корреляции должна быть в целом однородной. Систематические участки с "
                 "низкой корреляцией обычно указывают на проблемы в данных (мьютинг, следы "
                 "системы наблюдений, «мёртвые» трассы), а не в инверсии.",
        fig="s11_section.png",
        fig_en="The inverted impedance section. Compare it with the seismic from step 2: "
               "the same structure, but now readable as rock rather than as reflections.",
        fig_ru="Разрез полученного импеданса. Сравните его с сейсмическим разрезом из "
               "шага 2: структура та же, но теперь она читается как породы, а не как "
               "отражения.",
    ),
]

# --------------------------------------------------------------------------
# Methods, pitfalls, results, glossary
# --------------------------------------------------------------------------

METHOD_TABLE = {
    # Column widths are relative.  "Разреженно-импульсная" is 21 characters and
    # overflows a narrow first column, so it gets the room it needs.
    "head": [("Method / Метод", 41),
             ("Needs / Требует", 27),
             ("Gives / Даёт", 38),
             ("Use when / Когда применять", 54)],
    "rows": [
        ["Coloured\nЦветная",
         "No wavelet\nБез импульса",
         "Relative impedance\nОтносительный импеданс",
         "First pass over a whole volume; robust, seconds per survey.\n"
         "Первый проход по всему кубу; устойчива, секунды на съёмку."],
        ["Sparse-spike\nРазреженно-импульсная",
         "Wavelet\nИмпульс",
         "Sparse reflectivity\nРазреженные коэффициенты отражения",
         "Thin-bed detail, blocky geology. Most sensitive to tie quality.\n"
         "Детализация тонких пластов, блоковая геология. Наиболее чувствительна "
         "к привязке."],
        ["Model-based\nНа основе модели",
         "Wavelet + LFM\nИмпульс +\nнизкочастотная модель",
         "Absolute impedance\nАбсолютный импеданс",
         "The standard workhorse when the tie and the background model are good.\n"
         "Стандартный рабочий метод при хорошей привязке и низкочастотной модели."],
        ["Bayesian\nБайесовская",
         "Wavelet + LFM\nИмпульс +\nнизкочастотная модель",
         "Absolute impedance + uncertainty\nАбсолютный импеданс + неопределённость",
         "When you need P10/P90, realisations, or a probability on a cut-off.\n"
         "Когда нужны P10/P90, реализации или вероятность превышения предела."],
    ],
}

MULTIATTRIBUTE = [
    ("h1", "Predicting a log from the seismic",
     "Прогноз кривой ГИС по сейсмическим данным"),
    ("p",
     "The transform on step 10 goes through impedance, so it can only carry what the "
     "inversion resolved. The second route on that page goes straight from the seismic to "
     "the log: the classical multi-attribute method (Hampson, Schultz &amp; Fehler, 2001). "
     "It can find relations impedance alone misses — and it can just as easily fit noise, "
     "which is why the whole method is built around one safeguard.",
     "Зависимость на шаге 10 проходит через импеданс и поэтому несёт только то, что "
     "восстановила инверсия. Второй путь на этой странице ведёт от сейсмических данных к "
     "кривой ГИС напрямую — это классический многоатрибутный метод (Hampson, Schultz &amp; "
     "Fehler, 2001). Он способен уловить связи, недоступные одному импедансу, и с той же "
     "лёгкостью подстроиться под шум, поэтому весь метод построен вокруг одной защитной "
     "меры."),
    ("p",
     "Thirteen attributes are derived from the volume — amplitude, envelope, instantaneous "
     "phase and frequency, amplitude-weighted frequency, the integrated trace, the first and "
     "second derivatives, four band-passed copies and time itself — and, when a full-volume "
     "inversion exists, its impedance joins them. Each is read through a short "
     "<b>operator</b> spanning several samples rather than at a single sample: a log responds "
     "to a bed, a seismic sample responds to everything within a wavelet of it, and the two "
     "only line up over a window.",
     "Из куба рассчитываются тринадцать атрибутов: амплитуда, огибающая, мгновенная фаза и "
     "мгновенная частота, частота, взвешенная по амплитуде, интегрированная трасса, первая и "
     "вторая производные, четыре полосовые копии и само время; при наличии инверсии по всему "
     "кубу к ним добавляется её импеданс. Каждый атрибут берётся не в одной точке, а через "
     "короткий <b>оператор</b> длиной в несколько отсчётов: кривая ГИС отражает пласт, "
     "отсчёт сейсмической трассы — всё, что попадает в пределы импульса, и совместить их "
     "можно только на окне."),

    ("h2", "How the number of attributes is chosen", "Как выбирается число атрибутов"),
    ("numbered",
     ["Attributes are added one at a time. At each step every remaining attribute is tried, "
      "and the one that lowers the error most is kept.",
      "The error that decides is measured on a well the model has <b>not</b> seen: each well "
      "is held out in turn, the model is refitted without it, and scored against its log.",
      "Selection stops when that error stops improving — whatever the training error is "
      "doing."],
     ["Атрибуты добавляются по одному. На каждом шаге перебираются все оставшиеся атрибуты, "
      "и выбирается тот, который сильнее всего снижает ошибку.",
      "Решающая ошибка измеряется на скважине, которую модель <b>не видела</b>: каждая "
      "скважина по очереди исключается, модель настраивается заново без неё и сравнивается с "
      "её кривой.",
      "Отбор прекращается, когда эта ошибка перестаёт уменьшаться, — независимо от того, что "
      "происходит с ошибкой на обучающих данных."]),
    ("note",
     "Training error can only fall as attributes are added. It is not evidence of anything. "
     "A method that chose its size by training error would fit noise until it ran out of "
     "attributes — which is how multi-attribute prediction earned its reputation for "
     "producing beautiful maps of nothing.",
     "Ошибка на обучающих данных может только уменьшаться по мере добавления атрибутов и "
     "поэтому ничего не доказывает. Метод, выбирающий размер модели по ней, подстраивался бы "
     "под шум, пока не закончатся атрибуты, — именно так многоатрибутный прогноз заработал "
     "репутацию источника красивых карт, за которыми ничего нет."),
    ("figure", "s10_multi_error.png",
     "The panel that decides. Training error (blue) can only fall; validation error (red), "
     "measured on a well the fit never saw, is the one that matters, and the dotted line "
     "marks where selection stopped. The dashed line across the top is the error you would "
     "get by predicting the mean everywhere: a validation curve that never drops clearly "
     "below it means the seismic is not carrying this property, however good the training "
     "fit looks.",
     "Решающая панель. Ошибка на обучающих данных (синяя) может только убывать; значение "
     "имеет ошибка на валидационной скважине (красная), измеренная на данных, которых модель "
     "не видела, а пунктирная вертикаль отмечает момент остановки отбора. Горизонтальная "
     "штриховая линия — ошибка, которую дало бы простое предсказание среднего: если красная "
     "кривая не опускается заметно ниже неё, сейсмические данные не несут информации об этом "
     "свойстве, каким бы хорошим ни выглядело обучение.",
     1.0, 88),

    ("h2", "What the app says out loud", "О чём программа предупреждает прямо"),
    ("bullets",
     ["<b>Validation error no better than predicting the mean</b> — the model has learned "
      "nothing that generalises to a well it has not seen.",
      "<b>Validation error still falling at the attribute limit</b> — the limit chose the "
      "size of the model, not the data. Raise it and refit.",
      "<b>Validation error far above training error</b> — the extra freedom is being spent "
      "on memorising the wells.",
      "<b>Fewer than two wells</b> — the fit is refused outright: with one well there is "
      "nothing to validate against.",
      "<b>A neural network that validates worse than the linear model</b> — said plainly, "
      "rather than presenting the more elaborate answer as the better one."],
     ["<b>Ошибка на валидационной скважине не лучше предсказания среднего</b> — модель не "
      "выявила ничего, что переносится на скважину, которую она не видела.",
      "<b>Ошибка продолжала убывать на пределе числа атрибутов</b> — размер модели определён "
      "этим пределом, а не данными. Увеличьте предел и повторите настройку.",
      "<b>Ошибка на валидационной скважине намного больше ошибки на обучающих данных</b> — "
      "избыточная свобода тратится на запоминание скважин.",
      "<b>Менее двух скважин</b> — настройка отклоняется: по одной скважине проверять не на "
      "чем.",
      "<b>Нейронная сеть проверяется хуже линейной модели</b> — об этом сообщается прямо, а "
      "не подаётся как более совершенный результат."]),
    ("p",
     "The <b>neural network</b> option is one hidden layer, fitted on the attributes the "
     "linear selection already chose, so the two are compared on equal footing. The "
     "uncertainty carried into the cut-off is the validation error itself, held constant over "
     "the cube: there is no posterior on this route, and what the predictor missed at a well "
     "it had never seen is the only error bar it has earned.",
     "Вариант с <b>нейронной сетью</b> — один скрытый слой, настраиваемый на тех же "
     "атрибутах, которые отобрала линейная модель, чтобы сравнение было равным. В расчёт "
     "предела передаётся сама ошибка на валидационной скважине, постоянная по всему кубу: "
     "апостериорного распределения на этом пути нет, и то, что прогноз не угадал на скважине, "
     "которую не видел, — единственная обоснованная оценка его погрешности."),
    ("figure", "s10_multi_well.png",
     "The predicted curve against the measured log at one well, on the demo dataset. The "
     "final model is fitted on every well, so this panel shows the fit rather than a blind "
     "test — the blind number is the validation error above. What it does show honestly is "
     "the resolution on offer: the prediction follows the log where the seismic has "
     "something to say and smooths through the thin beds it cannot resolve.",
     "Прогнозная кривая и измеренный каротаж в одной скважине на демонстрационном наборе. "
     "Итоговая модель настроена по всем скважинам, поэтому панель показывает качество "
     "настройки, а не независимую проверку — независимая оценка приведена выше. Зато она "
     "честно показывает достижимую разрешающую способность: прогноз следует за каротажем "
     "там, где сейсмика несёт информацию, и сглаживает тонкие пласты, которые она не "
     "разрешает.",
     0.82, 96),
    ("note",
     "<b>Which route?</b> The transform is the safer default: one relation, one set of "
     "coefficients, and its failure is visible as a low R². Multi-attribute prediction is "
     "worth reaching for when impedance demonstrably does not explain the curve, and when "
     "there are enough wells that holding one out still leaves a model worth fitting. Two "
     "wells make validation possible; five or more make it meaningful.",
     "<b>Какой путь выбрать?</b> Зависимость через импеданс безопаснее как выбор по "
     "умолчанию: одно соотношение, один набор коэффициентов, а его несостоятельность видна по "
     "низкому R². Многоатрибутный прогноз оправдан, когда импеданс явно не объясняет кривую и "
     "когда скважин достаточно, чтобы после исключения одной модель оставалась осмысленной. "
     "Две скважины делают проверку возможной, пять и более — содержательной."),
]

PITFALLS = [
    ("h1", "Six things that go wrong", "Шесть типичных ошибок"),
    ("numbered",
     [("<b>Units.</b> A sonic read as µs/m when it is really µs/ft scales velocity by 3.28. "
       "A density read as g/cm³ when it is kg/m³ is out by 1,000. A depth index in feet "
       "read as metres puts the whole well 3.28× too deep. All three produce impedance "
       "that is confidently, silently wrong. Step 3 exists to catch them."),
      ("<b>Wavelet amplitude.</b> A peak-normalised wavelet is what you want for display "
       "and is the wrong operator for inversion: if the wavelet is off by a factor, every "
       "reflection coefficient is off by its inverse. On real data the calibration factor "
       "was over 8,000. The app calibrates automatically — do not bypass it."),
      ("<b>The analysis gate.</b> Set it to the interval where the logs actually have both "
       "sonic and density. The two tools rarely start at the same depth, and scoring a tie "
       "over an interval where the well contributes nothing drags the answer toward "
       "whatever the seismic happens to do there."),
      ("<b>Non-blind scoring.</b> With one well, the background model is built from the same "
       "log you are scoring against, so the full-band correlation mostly measures the "
       "background. Quote the band above the model's cutoff, or use step 9."),
      ("<b>Under-regularised sparse-spike.</b> Driving the residual to near zero means "
       "fitting noise. On real data an under-regularised run scored <i>below</i> the "
       "background model while fitting the seismic to 0.5%. Choose the weight from the "
       "noise level."),
      ("<b>A training fit is not a result.</b> On the multi-attribute route, adding "
       "attributes always improves the fit at the wells the model was given. Only the error "
       "at a well held out of the fit says whether anything was learned — and it means "
       "nothing until it is compared with the spread of the curve itself. An error equal to "
       "the spread is a model no better than predicting the mean.")],
     [("<b>Единицы измерения.</b> Акустический каротаж, прочитанный как мкс/м вместо "
       "мкс/фут, завышает скорость в 3,28 раза. Плотность, прочитанная как г/см³ вместо "
       "кг/м³, ошибочна в 1 000 раз. Глубина в футах, принятая за метры, смещает всю "
       "скважину в 3,28 раза глубже. Все три случая дают уверенно и незаметно неверный "
       "импеданс. Шаг 3 предназначен именно для их выявления."),
      ("<b>Амплитуда импульса.</b> Импульс, нормированный по максимуму, удобен для "
       "отображения, но непригоден как оператор инверсии: если импульс масштабирован "
       "неверно, все коэффициенты отражения искажаются обратно пропорционально. На "
       "реальных данных калибровочный коэффициент превысил 8 000. Программа калибрует "
       "автоматически — не отключайте это."),
      ("<b>Интервал анализа.</b> Задавайте его по интервалу, где реально есть и "
       "акустический, и плотностной каротаж. Эти приборы редко начинают запись с одной "
       "глубины, а оценка привязки на интервале, где скважина ничего не даёт, смещает "
       "результат в сторону случайного поведения сейсмики."),
      ("<b>Оценка по использованным данным.</b> При одной скважине низкочастотная модель "
       "строится по той же кривой, с которой сравнивается результат, поэтому "
       "широкополосная корреляция оценивает в основном саму модель. Приводите корреляцию "
       "в полосе выше частоты среза модели или используйте шаг 9."),
      ("<b>Недорегуляризованная разреженно-импульсная инверсия.</b> Сведение невязки почти "
       "к нулю означает подстройку под шум. На реальных данных такой расчёт дал результат "
       "<i>хуже</i> низкочастотной модели при невязке 0,5 %. Подбирайте вес по уровню "
       "шума."),
      ("<b>Совпадение на обучающих данных — не результат.</b> На многоатрибутном пути "
       "добавление атрибутов всегда улучшает совпадение на тех скважинах, которые были даны "
       "модели. О том, выявлено ли что-либо, говорит только ошибка на исключённой скважине — "
       "и она ничего не значит, пока не сопоставлена с разбросом самой кривой. Ошибка, "
       "равная разбросу, означает модель не лучше предсказания среднего.")]),
]

RESULTS = [
    ("h1", "How well does it actually work?", "Насколько это работает на практике"),
    ("p",
     "Tested on the <b>Penobscot 3D</b> survey, offshore Nova Scotia — real seismic and a "
     "real well (L-30), both openly licensed. The well sits 42 m from the line. Scores are "
     "the correlation with the well log in the 10–60 Hz band, above the background model's "
     "cutoff, so the background carries no information there and the number reflects what "
     "the inversion actually recovered from the seismic.",
     "Проверка выполнена на съёмке <b>Penobscot 3D</b> (шельф Новой Шотландии) — реальные "
     "сейсмические данные и реальная скважина L-30, обе в открытом доступе. Скважина "
     "расположена в 42 м от профиля. Приведена корреляция с кривой ГИС в полосе 10–60 Гц, "
     "выше частоты среза низкочастотной модели: в этой полосе модель не несёт информации, "
     "поэтому число отражает то, что инверсия действительно извлекла из сейсмики."),
    ("figure", "penobscot.png",
     "Real data, all four methods. Left: the recorded seismic. Centre: Bayesian absolute "
     "impedance — note the fault at inline 1220 and the high-impedance package below "
     "2500 ms. Right: the 10–60 Hz band at the well, where the background model carries "
     "nothing and the inversion has to earn its score.",
     "Реальные данные, все четыре метода. Слева: записанная сейсмика. В центре: "
     "абсолютный импеданс по байесовской инверсии — обратите внимание на разлом около "
     "профиля 1220 и высокоимпедансную пачку ниже 2500 мс. Справа: полоса 10–60 Гц в "
     "точке скважины, где низкочастотная модель не несёт информации и результат "
     "определяется только инверсией.",
     1.0, 96),
    ("resulttable", None, None),
    ("note",
     "<b>The tie dominates everything.</b> No engine changed between these two columns — "
     "only the well tie did. L-30 has no checkshot, so its time-depth comes from "
     "integrating a sonic, and that drifts. Allowing a stretch on top of the bulk shift "
     "was worth <b>+0.330</b> of tie correlation, and every engine rose with it. On data "
     "without a checkshot, tie quality is worth more than the choice of method.",
     "<b>Привязка определяет всё.</b> Между этими двумя столбцами не менялся ни один "
     "алгоритм — менялась только привязка скважины. У L-30 нет данных ВСП, поэтому "
     "зависимость время–глубина получена интегрированием акустического каротажа и "
     "накапливает расхождение. Добавление растяжения поверх постоянного сдвига дало "
     "<b>+0,330</b> к корреляции привязки, и вместе с ней выросли все алгоритмы. На "
     "данных без ВСП качество привязки важнее выбора метода."),
]

GLOSSARY = [
    ("Acoustic impedance (AI)", "Акустический импеданс",
     "Velocity × density. The property the inversion solves for.",
     "Скорость × плотность. Величина, которую восстанавливает инверсия."),
    ("Post-stack", "Суммарные данные",
     "Stacked (summed) seismic; no angle information.",
     "Суммированные сейсмические данные; информация об углах отсутствует."),
    ("Reflectivity", "Коэффициенты отражения",
     "Contrast in impedance between two layers.",
     "Контраст импеданса между двумя слоями."),
    ("Wavelet", "Сейсмический импульс",
     "The pulse the earth was illuminated with.",
     "Импульс, которым была «освещена» среда."),
    ("Well tie", "Привязка скважины",
     "Matching the well's synthetic to the seismic in time.",
     "Совмещение синтетической сейсмограммы скважины с сейсмикой во времени."),
    ("Bulk shift", "Постоянный сдвиг",
     "Moving the whole well in time by a constant.",
     "Смещение всей скважины во времени на постоянную величину."),
    ("Stretch and squeeze", "Растяжение и сжатие",
     "Correcting a time-depth that drifts with depth.",
     "Исправление зависимости время–глубина, расходящейся с глубиной."),
    ("Checkshot", "Данные ВСП",
     "Measured time-depth relationship. Preferred over integrating a sonic.",
     "Измеренная зависимость время–глубина. Предпочтительнее интегрирования "
     "акустического каротажа."),
    ("Deviation survey", "Инклинометрия",
     "The borehole's path in X, Y and TVD against measured depth.",
     "Траектория ствола в координатах X, Y и по вертикали в зависимости от глубины "
     "по стволу."),
    ("Low-frequency model", "Низкочастотная модель",
     "Background impedance below the seismic band.",
     "Фоновый импеданс ниже полосы частот сейсмики."),
    ("Analysis gate", "Интервал анализа",
     "The time window used for wavelet extraction and scoring.",
     "Временное окно для извлечения импульса и оценки качества."),
    ("Misfit / residual", "Невязка",
     "What the model failed to explain in the seismic.",
     "То, что модель не смогла объяснить в сейсмических данных."),
    ("Prior / posterior", "Априорный / апостериорный",
     "Belief before and after the seismic is taken into account.",
     "Представление до и после учёта сейсмических данных."),
    ("P10 / P90", "P10 / P90",
     "Low and high estimates bracketing 80% of the posterior.",
     "Нижняя и верхняя оценки, охватывающие 80 % апостериорного распределения."),
    ("Blind well", "Валидационная скважина",
     "A well held out of the model and used only for scoring.",
     "Скважина, исключённая из модели и используемая только для оценки."),
    ("Uplift", "Улучшение",
     "How much the inversion improved on its own background model.",
     "Насколько инверсия улучшила результат по сравнению со своей низкочастотной "
     "моделью."),
    ("Q factor", "Поглощение",
     "How strongly the earth absorbs high frequencies.",
     "Насколько сильно среда поглощает высокие частоты."),
    ("Net thickness", "Эффективная толщина",
     "Thickness passing a property cut-off.",
     "Толщина, удовлетворяющая заданному пределу по свойству."),
    ("Inline / crossline", "Продольный / поперечный профиль",
     "The two grid directions of a 3D survey.",
     "Два направления сетки съёмки 3D."),
    ("Formation tops", "Отбивки",
     "Picked formation boundaries in a well.",
     "Отмеченные границы пластов в скважине."),
    ("Seismic attribute", "Сейсмический атрибут",
     "A quantity derived from the trace: envelope, phase, frequency and so on.",
     "Величина, рассчитанная по трассе: огибающая, фаза, частота и подобные."),
    ("Multi-attribute prediction", "Многоатрибутный прогноз",
     "Predicting a well curve from several seismic attributes at once.",
     "Прогноз кривой ГИС сразу по нескольким сейсмическим атрибутам."),
    ("Training error", "Ошибка на обучающих данных",
     "Error at the wells the model was fitted on. Always falls as the model grows.",
     "Ошибка на скважинах, по которым настроена модель. Всегда убывает с ростом модели."),
    ("Validation error", "Ошибка на валидационной скважине",
     "Error at a well held out of the fit. The number that decides.",
     "Ошибка на скважине, исключённой из настройки. Именно она является решающей."),
    ("Overfitting", "Переобучение",
     "Fitting the noise in the wells instead of the relation between them.",
     "Подстройка под шум в скважинах вместо связи, общей для них."),
    ("Operator length", "Длина оператора",
     "How many samples of an attribute are used to predict one sample of the log.",
     "Сколько отсчётов атрибута используется для прогноза одного отсчёта кривой."),
]
