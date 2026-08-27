"""Bilingual content for the post-stack inversion user guide.

Each entry is (kind, english, russian).  Kept apart from the layout code so the
wording can be revised without touching the PDF machinery.
"""

TITLE_EN = "Post-Stack Seismic Inversion"
TITLE_RU = "Постстековая сейсмическая инверсия"
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
     "Программа преобразует куб суммарных (постстековых) сейсмических данных в куб "
     "<b>акустического импеданса</b> — физического свойства породы, калиброванного "
     "по данным ГИС. Разрезы импеданса читаются как геология, в отличие от разрезов "
     "амплитуд: яркая амплитуда может быть эффектом настройки, а контраст импеданса — "
     "это реальное изменение породы."),
    ("p",
     "Four inversion engines are included, plus the QC needed to trust the result: "
     "well-tie optimisation, wavelet estimation, a low-frequency model, blind-well "
     "cross-validation, and uncertainty. The workflow is eleven steps and is meant to "
     "be followed in order — each step consumes what the previous one produced.",
     "Реализованы четыре алгоритма инверсии и весь контроль качества, необходимый, "
     "чтобы результату можно было доверять: оптимизация увязки скважин, оценка "
     "сейсмического импульса, низкочастотная модель, слепая кросс-валидация и оценка "
     "неопределённости. Рабочий процесс состоит из одиннадцати шагов и выполняется "
     "по порядку: каждый шаг использует результат предыдущего."),

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
     ["Куб <b>постстековых данных SEG-Y</b> 3D (или 2D-линия). До 1 ГБ через браузер.",
      "Хотя бы одна <b>скважина</b> с акустическим (DT) и плотностным (RHOB) каротажем "
      "в формате LAS.",
      "<b>Координаты X/Y</b> скважин — из заголовка LAS или из CSV-файла заголовков. "
      "Без них скважину нельзя привязать к сейсмике, и она будет проигнорирована.",
      "Желательно: <b>сейсмокаротаж</b> (связь время–глубина), <b>инклинометрия</b>, "
      "<b>отбивки</b> пластов и интерпретированные <b>горизонты</b> в формате CSV.",
      "<b>Две скважины и более</b> — если нужна слепая проверка: по одной скважине "
      "слепую оценку получить невозможно."]),

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
     "раннем шаге (скважины, увязка, куб) намеренно очищает все зависимые результаты, "
     "чтобы на экране не осталось устаревшего результата."),
]

# --------------------------------------------------------------------------
# The eleven steps.  Each: number, title, what it does, what you do, what to check
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
        ru_do="Выберите источник. Для SEG-Y проверьте байтовые позиции inline / crossline / "
              "X / Y по таблице заголовков, которую показывает программа. Если в LAS нет "
              "координат, загрузите CSV с заголовками скважин.",
        en_check="Trace count, inline and crossline ranges, sample rate and time range are "
                 "plausible. Every well you expect appears, and each has an X/Y.",
        ru_check="Количество трасс, диапазоны inline и crossline, шаг дискретизации и "
                 "временной диапазон правдоподобны. Все ожидаемые скважины загружены и у "
                 "каждой есть координаты X/Y.",
    ),
    dict(
        n=2,
        en_title="Seismic viewer", ru_title="Просмотр сейсмики",
        en_does="Inline, crossline, time slice, and an arbitrary traverse through chosen "
                "wells in a chosen order, with gain, clip and colour controls and well "
                "overlays.",
        ru_does="Просмотр inline, crossline, временного среза и произвольного профиля через "
                "выбранные скважины в заданном порядке, с регулировкой усиления, "
                "ограничения и цветовой шкалы, с отображением скважин.",
        en_do="Look at the data before inverting it. Check polarity, the mute, dead traces "
              "and any acquisition footprint.",
        ru_do="Просмотрите данные до инверсии. Проверьте полярность, зону мьютинга, "
              "«мёртвые» трассы и следы системы наблюдений.",
        en_check="The wells land where you expect on the sections. If a well sits on a dead "
                 "trace or outside the survey, fix it here rather than later.",
        ru_check="Скважины расположены на разрезах там, где ожидается. Если скважина "
                 "попадает на «мёртвую» трассу или вне съёмки, исправьте это сейчас.",
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
    ),
    dict(
        n=4,
        en_title="Well correlation", ru_title="Корреляция скважин",
        en_does="Wells side by side in an order you choose, with logs, formation tops, the "
                "seismic trace at each well, correlation lines between tops, and optional "
                "flattening on a datum.",
        ru_does="Скважины рядом друг с другом в выбранном порядке, с кривыми ГИС, отбивками "
                "пластов, сейсмической трассой у каждой скважины, линиями корреляции между "
                "отбивками и возможностью выравнивания на выбранный репер.",
        en_do="Set the well order and pick which curves and tops to show. Flatten on a "
              "marker to compare thickness rather than depth.",
        ru_do="Задайте порядок скважин и выберите отображаемые кривые и отбивки. Выровняйте "
              "разрез на маркирующий горизонт, чтобы сравнивать толщины, а не глубины.",
        en_check="Tops are consistent from well to well, and the seismic character at each "
                 "well matches the logs. A top that jumps between wells is usually a "
                 "picking error, not geology.",
        ru_check="Отбивки согласованы между скважинами, а волновая картина у каждой "
                 "скважины соответствует кривым ГИС. Резкий скачок отбивки от скважины к "
                 "скважине — обычно ошибка корреляции, а не геология.",
    ),
    dict(
        n=5,
        en_title="Well tie QC", ru_title="Увязка скважин с сейсмикой",
        en_does="Compares the well's synthetic seismogram with the trace extracted at the "
                "well, scores the tie, and repairs it: a constant bulk shift, and a "
                "piecewise-linear stretch on top of it.",
        ru_does="Сравнивает синтетическую сейсмограмму по скважине с трассой, извлечённой "
                "в точке скважины, оценивает качество увязки и исправляет её: постоянный "
                "сдвиг и кусочно-линейное растяжение/сжатие поверх него.",
        en_do="Run <b>Optimise tie</b>. A bulk shift fixes a datum error; a stretch fixes a "
              "time-depth that drifts, which is what you get whenever the time-depth comes "
              "from integrating a sonic instead of a checkshot. For deviated wells, switch "
              "on <b>Follow the borehole path</b>.",
        ru_do="Запустите <b>Optimise tie</b> (оптимизация увязки). Постоянный сдвиг "
              "устраняет ошибку привязки к уровню приведения; растяжение устраняет "
              "накопленное расхождение зависимости время–глубина, возникающее всегда, когда "
              "она получена интегрированием акустического каротажа, а не сейсмокаротажем. "
              "Для наклонных скважин включите <b>Follow the borehole path</b> (по стволу).",
        en_check="Tie correlation. Above ~0.6 is workable; below ~0.4 everything downstream "
                 "will suffer. <b>This step matters more than the choice of inversion "
                 "method</b> — see the validation results at the end of this guide.",
        ru_check="Коэффициент корреляции увязки. Выше ~0,6 — приемлемо; ниже ~0,4 пострадают "
                 "все последующие шаги. <b>Этот шаг важнее выбора алгоритма инверсии</b> — "
                 "см. результаты проверки в конце руководства.",
    ),
    dict(
        n=6,
        en_title="Wavelet", ru_title="Сейсмический импульс (вейвлет)",
        en_does="Builds the wavelet: parametric (Ricker, Ormsby, Butterworth), statistical "
                "from the seismic, or least-squares extraction from the wells. Also "
                "estimates bulk Q and can extract a separate wavelet per time window.",
        ru_does="Формирование импульса: параметрический (Рикер, Ормсби, Баттерворт), "
                "статистический по сейсмике или извлечённый по скважинам методом наименьших "
                "квадратов. Также оценивается добротность Q и может извлекаться отдельный "
                "импульс для каждого временного окна.",
        en_do="Prefer <b>well-based extraction</b> — it is the only route that recovers the "
              "true phase of the data. Set the wavelet length to about twice the dominant "
              "period. Check the Q estimate to see whether one wavelet is enough.",
        ru_do="Предпочтительно <b>извлечение по скважинам</b> — только этот способ "
              "восстанавливает истинную фазу данных. Длину импульса задайте примерно в два "
              "раза больше преобладающего периода. Оцените Q, чтобы понять, достаточно ли "
              "одного импульса на весь куб.",
        en_check="Amplitude and phase spectra, and the tie correlation the wavelet achieves. "
                 "A frequency drift of a few Hz across the volume is normal; tens of Hz "
                 "means the deep section is being inverted with a wavelet it does not have.",
        ru_check="Амплитудный и фазовый спектры и достигнутая корреляция увязки. Смещение "
                 "частоты на несколько Гц по кубу нормально; десятки Гц означают, что "
                 "глубокая часть разреза инвертируется с неподходящим импульсом.",
    ),
    dict(
        n=7,
        en_title="Low-frequency model", ru_title="Низкочастотная модель",
        en_does="Builds the background impedance cube: well impedance low-pass filtered "
                "(default 10 Hz) and interpolated laterally between wells, optionally "
                "guided by horizons.",
        ru_does="Построение фонового куба импеданса: импеданс по скважинам, отфильтрованный "
                "ФНЧ (по умолчанию 10 Гц) и интерполированный между скважинами, при "
                "необходимости с учётом горизонтов.",
        en_do="Load horizons if you have them. <b>One horizon</b> gives a datum shift; "
              "<b>two or more</b> give proportional (layer-cake) flattening, which follows "
              "thickness variation and not just structure.",
        ru_do="Загрузите горизонты, если они есть. <b>Один горизонт</b> задаёт сдвиг к "
              "реперу; <b>два и более</b> дают пропорциональное выравнивание, учитывающее "
              "не только структуру, но и изменение толщин.",
        en_check="The model is smooth, positive everywhere, and matches each well's filtered "
                 "log at that well. Seismic cannot recover frequencies below about 8–10 Hz, "
                 "so this model — not the seismic — carries the absolute impedance level.",
        ru_check="Модель гладкая, всюду положительная и совпадает с отфильтрованной кривой в "
                 "точке каждой скважины. Сейсмика не восстанавливает частоты ниже ~8–10 Гц, "
                 "поэтому именно эта модель, а не сейсмика, задаёт абсолютный уровень "
                 "импеданса.",
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
    ),
    dict(
        n=9,
        en_title="Blind validation", ru_title="Слепая проверка",
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
              "строить фоновую модель.",
        en_check="The <b>uplift</b> column: the correlation the inversion added over its own "
                 "background at a well it had not seen. An engine that cannot beat its own "
                 "background has added nothing, however good the section looks.",
        ru_check="Столбец <b>uplift</b> (прирост): насколько инверсия улучшила результат по "
                 "сравнению с собственной фоновой моделью в «слепой» скважине. Алгоритм, не "
                 "превосходящий свою фоновую модель, не добавил ничего, каким бы красивым "
                 "ни выглядел разрез.",
    ),
    dict(
        n=10,
        en_title="Rock property", ru_title="Свойства породы",
        en_does="Fits a transform from log-impedance to a well curve (porosity, for "
                "example), applies it to the cube, and combines the inversion uncertainty "
                "with the scatter of the wells about the fit.",
        ru_does="Подбор зависимости между логарифмом импеданса и кривой ГИС (например, "
                "пористостью), применение её к кубу и объединение неопределённости инверсии "
                "с разбросом скважинных точек относительно подобранной зависимости.",
        en_do="Choose the curve and the polynomial degree, fit, then predict the cube. Set a "
              "cut-off to get a probability map and an expected net thickness.",
        ru_do="Выберите кривую и степень полинома, выполните подбор, затем рассчитайте куб. "
              "Задайте граничное значение, чтобы получить карту вероятности и ожидаемую "
              "эффективную толщину.",
        en_check="R² of the fit. Below about 0.3, impedance explains little of that curve and "
                 "the predicted cube will be closer to a constant than to a measurement. "
                 "Check the crossplot for one cloud per well — that means the relation is "
                 "well-specific and should not be applied across the survey.",
        ru_check="Коэффициент детерминации R². Ниже ~0,3 импеданс почти не объясняет эту "
                 "кривую, и расчётный куб будет ближе к константе, чем к измерению. На "
                 "кросс-плоте проверьте, не образует ли каждая скважина отдельное облако "
                 "точек: это означает, что зависимость индивидуальна для скважины и не "
                 "должна распространяться на всю площадь.",
    ),
    dict(
        n=11,
        en_title="Results & export", ru_title="Результаты и экспорт",
        en_does="Section viewer, time slices, per-trace QC maps, posterior uncertainty and "
                "P10/P90 volumes where available, crossplots against well logs, and export "
                "to SEG-Y or CSV.",
        ru_does="Просмотр разрезов, временных срезов, карт контроля качества по трассам, "
                "апостериорной неопределённости и кубов P10/P90 (если рассчитаны), "
                "кросс-плоты против кривых ГИС и экспорт в SEG-Y или CSV.",
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
         "Детализация тонких пластов, блоковая геология. Наиболее чувствительна к увязке."],
        ["Model-based\nНа основе модели",
         "Wavelet + LFM\nИмпульс + НЧ-модель",
         "Absolute impedance\nАбсолютный импеданс",
         "The standard workhorse when the tie and the background model are good.\n"
         "Стандартный рабочий метод при хорошей увязке и фоновой модели."],
        ["Bayesian\nБайесовская",
         "Wavelet + LFM\nИмпульс + НЧ-модель",
         "Absolute impedance + uncertainty\nАбсолютный импеданс + неопределённость",
         "When you need P10/P90, realisations, or a probability on a cut-off.\n"
         "Когда нужны P10/P90, реализации или вероятность превышения кондиции."],
    ],
}

PITFALLS = [
    ("h1", "Five things that go wrong", "Пять типичных ошибок"),
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
       "noise level.")],
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
       "глубины, а оценка увязки на интервале, где скважина ничего не даёт, смещает "
       "результат в сторону случайного поведения сейсмики."),
      ("<b>Неслепая оценка.</b> При одной скважине фоновая модель строится по той же "
       "кривой, с которой сравнивается результат, поэтому широкополосная корреляция "
       "оценивает в основном фоновую модель. Приводите корреляцию в полосе выше частоты "
       "среза модели или используйте шаг 9."),
      ("<b>Недорегуляризованная разреженно-импульсная инверсия.</b> Сведение невязки почти "
       "к нулю означает подстройку под шум. На реальных данных такой расчёт дал результат "
       "<i>хуже</i> фоновой модели при невязке 0,5 %. Подбирайте вес по уровню шума.")]),
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
     "выше частоты среза фоновой модели: в этой полосе фоновая модель не несёт информации, "
     "поэтому число отражает то, что инверсия действительно извлекла из сейсмики."),
    ("resulttable", None, None),
    ("note",
     "<b>The tie dominates everything.</b> No engine changed between these two columns — "
     "only the well tie did. L-30 has no checkshot, so its time-depth comes from "
     "integrating a sonic, and that drifts. Allowing a stretch on top of the bulk shift "
     "was worth <b>+0.330</b> of tie correlation, and every engine rose with it. On data "
     "without a checkshot, tie quality is worth more than the choice of method.",
     "<b>Увязка определяет всё.</b> Между этими двумя столбцами не менялся ни один "
     "алгоритм — менялась только увязка скважины. У L-30 нет сейсмокаротажа, поэтому "
     "зависимость время–глубина получена интегрированием акустического каротажа и "
     "накапливает расхождение. Добавление растяжения поверх постоянного сдвига дало "
     "<b>+0,330</b> к корреляции увязки, и вместе с ней выросли все алгоритмы. На данных "
     "без сейсмокаротажа качество увязки важнее выбора метода."),
]

GLOSSARY = [
    ("Acoustic impedance (AI)", "Акустический импеданс",
     "Velocity × density. The property the inversion solves for.",
     "Скорость × плотность. Величина, которую восстанавливает инверсия."),
    ("Post-stack", "Постстековые данные",
     "Stacked (summed) seismic; no angle information.",
     "Суммарные сейсмические данные; информация об углах отсутствует."),
    ("Reflectivity", "Коэффициенты отражения",
     "Contrast in impedance between two layers.",
     "Контраст импеданса между двумя слоями."),
    ("Wavelet", "Сейсмический импульс (вейвлет)",
     "The pulse the earth was illuminated with.",
     "Импульс, которым была «освещена» среда."),
    ("Well tie", "Увязка скважины",
     "Matching the well's synthetic to the seismic in time.",
     "Совмещение синтетической сейсмограммы скважины с сейсмикой во времени."),
    ("Bulk shift", "Постоянный сдвиг",
     "Moving the whole well in time by a constant.",
     "Смещение всей скважины во времени на постоянную величину."),
    ("Stretch and squeeze", "Растяжение и сжатие",
     "Correcting a time-depth that drifts with depth.",
     "Исправление зависимости время–глубина, расходящейся с глубиной."),
    ("Checkshot", "Сейсмокаротаж",
     "Measured time-depth relationship. Preferred over integrating a sonic.",
     "Измеренная зависимость время–глубина. Предпочтительнее интегрирования "
     "акустического каротажа."),
    ("Deviation survey", "Инклинометрия",
     "The borehole's path in X, Y and TVD against measured depth.",
     "Траектория ствола в координатах X, Y и по вертикали в зависимости от глубины "
     "по стволу."),
    ("Low-frequency model (LFM)", "Низкочастотная модель",
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
    ("Blind well", "Слепая скважина",
     "A well held out of the model and used only for scoring.",
     "Скважина, исключённая из модели и используемая только для оценки."),
    ("Uplift", "Прирост",
     "How much the inversion improved on its own background model.",
     "Насколько инверсия улучшила результат по сравнению со своей фоновой моделью."),
    ("Q factor", "Добротность Q",
     "How strongly the earth absorbs high frequencies.",
     "Насколько сильно среда поглощает высокие частоты."),
    ("Net thickness", "Эффективная толщина",
     "Thickness passing a property cut-off.",
     "Толщина, удовлетворяющая заданной кондиции по свойству."),
]
