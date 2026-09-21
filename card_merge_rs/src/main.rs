use std::env;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use filetime::FileTime;
use glob::glob;
use image::{imageops, Rgb, RgbImage};

/// `merge_<YYYYMMDD>_<count>.jpg`, dated from the first merged image and
/// suffixed if that name is taken. Matches CardMergeService in the Neddog Cards
/// app so a merge made here and one made there are named the same way.
fn output_filename(first: &Path, count: usize) -> PathBuf {
    let base = format!("merge_{}_{}", date_stamp(first), count);

    let mut candidate = PathBuf::from(format!("{}.jpg", base));
    let mut suffix = 2u32;
    while candidate.exists() {
        candidate = PathBuf::from(format!("{}-{}.jpg", base, suffix));
        suffix += 1;
    }
    candidate
}

/// Prefers the date the scanner wrote into the filename; falls back to the
/// file's modification date. (The app also checks EXIF in between, which these
/// scans do not carry.)
fn date_stamp(path: &Path) -> String {
    if let Some(from_name) = date_in_filename(path) {
        return from_name;
    }

    let secs = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or_else(|| {
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64
        });
    format_local_date(secs)
}

/// First digit-bounded, plausible YYYYMMDD run in the filename. A card named
/// `2023-Murray-Panini-Select-166` has a year but no such run, so it falls
/// through rather than having the year read as a date.
fn date_in_filename(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_string_lossy().to_string();
    let chars: Vec<char> = stem.chars().collect();

    for i in 0..chars.len().saturating_sub(7) {
        if !chars[i..i + 8].iter().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let bounded_left = i == 0 || !chars[i - 1].is_ascii_digit();
        let bounded_right = i + 8 == chars.len() || !chars[i + 8].is_ascii_digit();
        if !bounded_left || !bounded_right {
            continue;
        }
        let text: String = chars[i..i + 8].iter().collect();
        if is_plausible_date(&text) {
            return Some(text);
        }
    }
    None
}

fn is_plausible_date(yyyymmdd: &str) -> bool {
    if yyyymmdd.len() != 8 {
        return false;
    }
    let year: i32 = match yyyymmdd[0..4].parse() { Ok(v) => v, Err(_) => return false };
    let month: u32 = match yyyymmdd[4..6].parse() { Ok(v) => v, Err(_) => return false };
    let day: u32 = match yyyymmdd[6..8].parse() { Ok(v) => v, Err(_) => return false };
    (1900..=2999).contains(&year) && (1..=12).contains(&month) && (1..=31).contains(&day)
}

/// Local calendar date for a Unix timestamp, so a merge made late in the
/// evening is not stamped with tomorrow's UTC date.
fn format_local_date(secs: i64) -> String {
    unsafe {
        let t = secs as libc::time_t;
        let mut parts: libc::tm = std::mem::zeroed();
        if libc::localtime_r(&t, &mut parts).is_null() {
            return String::from("00000000");
        }
        format!(
            "{:04}{:02}{:02}",
            parts.tm_year + 1900,
            parts.tm_mon + 1,
            parts.tm_mday
        )
    }
}

// Returns the trailing integer before .jpg, e.g. "card_front003.jpg" -> Some(3)
fn extract_trailing_number(path: &Path) -> Option<u32> {
    let ext = path.extension()?.to_string_lossy().to_lowercase();
    if ext != "jpg" {
        return None;
    }
    let stem = path.file_stem()?.to_string_lossy();
    let digits: String = stem.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.chars().rev().collect::<String>().parse().ok()
}

fn average_perimeter_color(images: &[RgbImage]) -> Rgb<u8> {
    let mut sum = [0u64; 3];
    let mut count = 0u64;

    for img in images {
        let (w, h) = img.dimensions();
        let edges = (0..w).map(|x| (x, 0))           // top
            .chain((0..w).map(|x| (x, h - 1)))        // bottom
            .chain((0..h).map(|y| (0, y)))             // left
            .chain((0..h).map(|y| (w - 1, y)));        // right

        for (x, y) in edges {
            let p = img.get_pixel(x, y);
            sum[0] += p[0] as u64;
            sum[1] += p[1] as u64;
            sum[2] += p[2] as u64;
            count += 1;
        }
    }

    Rgb([
        (sum[0] / count) as u8,
        (sum[1] / count) as u8,
        (sum[2] / count) as u8,
    ])
}

fn paste(canvas: &mut RgbImage, img: &RgbImage, x: u32, y: u32) {
    imageops::overlay(canvas, img, x as i64, y as i64);
}

fn merge_images(image_files: &[PathBuf], output: PathBuf) -> PathBuf {
    let n = image_files.len();
    let images: Vec<RgbImage> = image_files
        .iter()
        .map(|f| {
            image::open(f)
                .unwrap_or_else(|e| panic!("Failed to open {}: {}", f.display(), e))
                .to_rgb8()
        })
        .collect();

    // Scanner output can vary by a few pixels. Size each grid cell to the
    // largest scan so no card edge is clipped.
    let width = images.iter().map(|img| img.width()).max().unwrap();
    let height = images.iter().map(|img| img.height()).max().unwrap();

    let merged = match n {
        2 => {
            let mut m = RgbImage::new(2 * width, height);
            for (i, img) in images.iter().enumerate() {
                paste(&mut m, img, i as u32 * width, 0);
            }
            m
        }
        3 => {
            let mut m = RgbImage::new(3 * width, height);
            for (i, img) in images.iter().enumerate() {
                paste(&mut m, img, i as u32 * width, 0);
            }
            m
        }
        5 => {
            let fill = average_perimeter_color(&images);
            let mut m = RgbImage::from_pixel(3 * width, 2 * height, fill);
            for i in 0..3usize {
                paste(&mut m, &images[i], i as u32 * width, 0);
            }
            for i in 0..2usize {
                paste(&mut m, &images[i + 3], i as u32 * width + width / 2, height);
            }
            m
        }
        _ => {
            let cols = ((n as f64) / 2.0).round() as u32;
            let mut m = RgbImage::new(cols * width, 2 * height);
            for (idx, img) in images.iter().enumerate() {
                let row = idx as u32 / cols;
                let col = idx as u32 % cols;
                paste(&mut m, img, col * width, row * height);
            }
            m
        }
    };

    merged
        .save(&output)
        .unwrap_or_else(|e| panic!("Failed to save {}: {}", output.display(), e));
    println!("Merged image saved as {}", output.display());
    output
}

fn touch_files_in_order(card_files: &[PathBuf], merged_file: &Path) {
    let base_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let ft = FileTime::from_unix_time(base_secs, 0);
    filetime::set_file_times(merged_file, ft, ft).expect("Failed to set merged file time");

    for (i, path) in card_files.iter().enumerate() {
        let ts = base_secs - (i as i64 + 1);
        let ft = FileTime::from_unix_time(ts, 0);
        if let Err(e) = filetime::set_file_times(path, ft, ft) {
            eprintln!("Warning: couldn't set time on {}: {}", path.display(), e);
        }
    }
}

fn main() {
    let mut args: Vec<String> = env::args().skip(1).collect();

    let use_even = args.contains(&"-e".to_string());
    let use_all = args.contains(&"-a".to_string());
    if use_even && use_all {
        panic!("Use either -e (even files) or -a (all files), not both");
    }
    args.retain(|a| a != "-e" && a != "-a");

    let patterns = if args.is_empty() {
        vec!["card*".to_string()]
    } else {
        args
    };

    let mut files: Vec<PathBuf> = Vec::new();
    for pattern in &patterns {
        if pattern.contains('*') || pattern.contains('?') {
            match glob(pattern) {
                Ok(entries) => files.extend(entries.flatten()),
                Err(e) => eprintln!("Invalid pattern '{}': {}", pattern, e),
            }
        } else {
            files.push(PathBuf::from(pattern));
        }
    }

    // Pair each file with its trailing number; drop files with no number
    let mut file_nums: Vec<(PathBuf, u32)> = files
        .into_iter()
        .filter_map(|f| extract_trailing_number(&f).map(|n| (f, n)))
        .collect();

    file_nums.sort_by_key(|(_, n)| *n);

    let filtered: Vec<PathBuf> = if use_all {
        file_nums.iter().map(|(f, _)| f.clone()).collect()
    } else {
        let parity: u32 = if use_even { 0 } else { 1 };
        file_nums
            .iter()
            .filter(|(_, n)| n % 2 == parity)
            .map(|(f, _)| f.clone())
            .collect()
    };

    let all_sorted: Vec<PathBuf> = file_nums.iter().map(|(f, _)| f.clone()).collect();

    if filtered.is_empty() {
        eprintln!("No images to merge (nothing matched, or every match was filtered out).");
        std::process::exit(1);
    }

    println!("Merging files: {:?}", filtered);
    let output = output_filename(&filtered[0], filtered.len());
    let merged = merge_images(&filtered, output);
    touch_files_in_order(&all_sorted, &merged);
}
