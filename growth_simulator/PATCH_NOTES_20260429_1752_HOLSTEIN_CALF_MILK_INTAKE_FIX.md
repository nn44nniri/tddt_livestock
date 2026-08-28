# PATCH NOTES — 2026-04-29 17:52 — Holstein calf milk-availability configuration fix

## هدف
رفع خطای منطقی در پیکربندی `breed=6` برای Holstein / Holstein-Friesian بدون تغییر در کد اجرایی شبیه‌ساز.

در نسخه قبلی، پارامتر `milk_available_a` در بردارهای Holstein (`LIBRARY60` و `LIBRARY61`) برابر `18.0` بود. این مقدار برای «ظرفیت تولید شیر گاو هلشتاین» قابل فهم است، اما در LiGAPS این خانه از بردار برای **milk available for calf** در محاسبه شروع رشد/خوراک گوساله استفاده می‌شود. در نتیجه گوساله Holstein تا حوالی `WEANINGTIME = 210` روز با شیر کافی پشتیبانی می‌شد و مصرف علوفه/خوراک جامد تا بیش از 200 روز صفر باقی می‌ماند.

## اصلاح انجام‌شده
فقط فایل پیکربندی زیر تغییر کرد:

- `config/settings.json`

تغییر دقیق:

- `breed_sex_libraries.LIBRARY60[14]`: از `18.0` به `8.0`
- `breed_sex_libraries.LIBRARY61[14]`: از `18.0` به `8.0`

توجه: index بالا صفرمبنا است؛ در مستندات LiGAPS این همان پارامتر ۱۵ بردار breed/sex یعنی `milk available A` است.

## منطق علمی و فنی
- `WEANINGTIME` عمومی مدل همچنان `210` روز باقی ماند تا منطق اصلی LiGAPS-Beef تغییر نکند.
- در نژادهای اصلی Bos taurus مثل Charolais، مقدار `milk available A` برابر `8` است و با وجود `WEANINGTIME = 210`، خوراک/علوفه در روزهای اولیه رشد وارد مصرف می‌شود.
- برای Holstein، مقدار قبلی `18` عملاً «پتانسیل بالای lactation گاو dairy» را به «شیر در دسترس گوساله» تبدیل کرده بود و به همین دلیل مصرف خوراک جامد به‌صورت غیرطبیعی تا پایان دوره شیرخوارگی مدل صفر می‌ماند.
- مقدار `8.0` به مقیاس پایه LiGAPS برای calf milk availability برگشت داده شد. این کار فقط پیکربندی breed=6 را اصلاح می‌کند و هیچ معادله، حلقه، API، export، یا منطق thermoregulation/feed/energy را تغییر نمی‌دهد.

## فایل‌های مرتبط
- `config/settings.json`
- `doc/HOLSTEIN_BREED6.md`
- `doc/BREEDS_AND_HOLSTEIN.md`
- `config/SETTINGS.md`

## وضعیت اعتبارسنجی
- اعتبارسنجی JSON روی `config/settings.json` انجام شد.
- بررسی `py_compile` روی فایل‌های اصلی انجام شد.
- تغییر کد اجرایی انجام نشد؛ بنابراین این patch از نوع configuration-only است.
