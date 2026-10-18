// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! Path rendering support for the software renderer using zeno

use super::PhysicalRect;
use super::draw_functions::{PremultipliedRgbaColor, TargetPixel};
use alloc::vec::Vec;
use zeno::{Cap, Fill, Join, Mask, Stroke, Style};

pub use zeno::Command;

/// Convert Slint's PathDataIterator to zeno's Command format
pub fn convert_path_data_to_zeno(
    path_data: i_slint_core::graphics::PathDataIterator,
    rotation: crate::RotationInfo,
    scale_factor: i_slint_core::lengths::ScaleFactor,
    offset: euclid::Vector2D<f32, i_slint_core::lengths::PhysicalPx>,
) -> Vec<Command> {
    use crate::Transform as _;
    use i_slint_core::lengths::LogicalPx;
    use lyon_path::Event;
    let mut commands = Vec::new();

    let convert_point = |p| {
        let p = (euclid::Point2D::<f32, LogicalPx>::from_untyped(p) * scale_factor + offset)
            .transformed(rotation);
        zeno::Point::new(p.x, p.y)
    };

    for event in path_data.iter() {
        match event {
            Event::Begin { at } => {
                commands.push(Command::MoveTo(convert_point(at)));
            }
            Event::Line { to, .. } => {
                commands.push(Command::LineTo(convert_point(to)));
            }
            Event::Quadratic { ctrl, to, .. } => {
                commands.push(Command::QuadTo(convert_point(ctrl), convert_point(to)));
            }
            Event::Cubic { ctrl1, ctrl2, to, .. } => {
                commands.push(Command::CurveTo(
                    convert_point(ctrl1),
                    convert_point(ctrl2),
                    convert_point(to),
                ));
            }
            Event::End { close, .. } => {
                if close {
                    commands.push(Command::Close);
                }
            }
        }
    }

    commands
}

/// patched: largest mask of a band, a path is rasterized in bands of at most this size
/// (kept permanently, see `PathBuffers`, so it is smaller than a mask for a whole band would be)
const MAX_MASK_BYTES: usize = 8 * 1024;

/// patched: buffers for rendering paths, allocated with the first path and then kept for all
/// paths. Allocating them for every path (the mask of a band, and the rasterizer cells of zeno)
/// needs free blocks every frame. On an MCU the heap becomes fragmented when components are
/// created and destroyed, and such an allocation then fails although enough memory is free in
/// total.
struct PathBuffers {
    mask: Vec<u8>,
    scratch: zeno::Scratch,
    // outline of a stroke, it grows to the largest stroked path and is then kept
    outline: Vec<Command>,
    // the subpaths of the path being rendered, see `BandCommands`
    subpaths: Vec<Subpath>,
}

impl Default for PathBuffers {
    fn default() -> Self {
        Self {
            mask: alloc::vec![0; MAX_MASK_BYTES],
            scratch: zeno::Scratch::new(),
            outline: Vec::new(),
            subpaths: Vec::new(),
        }
    }
}

/// patched: the commands of a subpath (from a MoveTo to the next) and its vertical extent, the
/// smallest and largest y of its points and control points (the box of a curve lies within
/// them), in the coordinates of the path
#[derive(Clone)]
struct Subpath {
    commands: core::ops::Range<usize>,
    top: f32,
    bottom: f32,
}

/// patched: collects the subpaths of `commands`
fn find_subpaths(commands: &[Command], subpaths: &mut Vec<Subpath>) {
    subpaths.clear();
    let extend = |subpaths: &mut Vec<Subpath>, index: usize, y: f32| {
        if let Some(last) = subpaths.last_mut() {
            last.commands.end = index + 1;
            last.top = last.top.min(y);
            last.bottom = last.bottom.max(y);
        }
    };
    for (index, command) in commands.iter().enumerate() {
        match *command {
            Command::MoveTo(p) => {
                subpaths.push(Subpath { commands: index..index + 1, top: p.y, bottom: p.y })
            }
            Command::LineTo(p) => extend(subpaths, index, p.y),
            Command::QuadTo(c, p) => {
                extend(subpaths, index, c.y);
                extend(subpaths, index, p.y);
            }
            Command::CurveTo(c1, c2, p) => {
                extend(subpaths, index, c1.y);
                extend(subpaths, index, c2.y);
                extend(subpaths, index, p.y);
            }
            // a Close has no point, min and max ignore NaN
            Command::Close => extend(subpaths, index, f32::NAN),
        }
    }
    // commands before the first MoveTo: no subpaths, the whole path is used in every band
    if subpaths.first().is_none_or(|first| first.commands.start != 0) {
        subpaths.clear();
        subpaths.push(Subpath {
            commands: 0..commands.len(),
            top: f32::NEG_INFINITY,
            bottom: f32::INFINITY,
        });
    }
}

/// patched: the commands of the subpaths that reach into the rows `top..bottom` of a band. A
/// path is rasterized band by band, and zeno processes all its commands in every band: for a
/// path of many small subpaths (the ticks on the card of the heading indicator) most of that
/// work is for subpaths that lie completely above or below the band, they have no edges in its
/// rows and don't change its coverage.
#[derive(Clone, Copy)]
struct BandCommands<'a> {
    commands: &'a [Command],
    subpaths: &'a [Subpath],
    top: f32,
    bottom: f32,
}

impl Subpath {
    fn reaches(&self, top: f32, bottom: f32) -> bool {
        self.bottom >= top && self.top <= bottom
    }
}

#[derive(Clone)]
struct BandCommandsIter<'a> {
    band: BandCommands<'a>,
    subpaths: core::slice::Iter<'a, Subpath>,
    current: core::slice::Iter<'a, Command>,
}

impl Iterator for BandCommandsIter<'_> {
    type Item = Command;
    fn next(&mut self) -> Option<Command> {
        loop {
            if let Some(command) = self.current.next() {
                return Some(*command);
            }
            let subpath = self.subpaths.next()?;
            if subpath.reaches(self.band.top, self.band.bottom) {
                self.current = self.band.commands[subpath.commands.clone()].iter();
            }
        }
    }
}

impl<'a> zeno::PathData for BandCommands<'a> {
    type Commands = BandCommandsIter<'a>;
    fn commands(&self) -> Self::Commands {
        BandCommandsIter { band: *self, subpaths: self.subpaths.iter(), current: [].iter() }
    }
}

i_slint_core::thread_local! {
    // patched: the kept path buffers, like the glyph cache of the renderer
    static PATH_BUFFERS: core::cell::RefCell<PathBuffers> = core::cell::RefCell::default();
}

/// patched: blends the color with the coverage of a pixel of the mask, `opaque` is the pixel of
/// an opaque color, which covers a pixel completely
#[inline(always)]
fn blend_coverage<T: TargetPixel>(
    pixel: &mut T,
    coverage: u8,
    color: PremultipliedRgbaColor,
    opaque: Option<T>,
) {
    match coverage {
        0 => {}
        255 => match opaque {
            Some(opaque) => *pixel = opaque,
            None => T::blend(pixel, color),
        },
        _ => {
            // Scale all color components by coverage to maintain premultiplication
            // (v + (v >> 8) + 1) >> 8 equals v / 255 for all products of two u8,
            // without a division, which is slow on MCUs like the Cortex-M0+
            let scale = |component: u8| {
                let v = component as u32 * coverage as u32;
                ((v + (v >> 8) + 1) >> 8) as u8
            };
            let alpha_color = PremultipliedRgbaColor {
                red: scale(color.red),
                green: scale(color.green),
                blue: scale(color.blue),
                alpha: scale(color.alpha),
            };
            T::blend(pixel, alpha_color);
        }
    }
}

/// Common rendering logic for both filled and stroked paths
///
/// patched: only the part of the path inside `clip_rects` (the non-overlapping parts of the dirty
/// region inside the clip) is rasterized, in horizontal bands with a mask of at most
/// `MAX_MASK_BYTES`. This bounds the memory on MCUs, avoids rasterizing pixels that are not
/// redrawn, and pixels outside the redrawn region are not blended again.
/// patched: the mask and the rasterizer cells come from `buffers`, see `PATH_BUFFERS`.
fn render_path_with_style<T: TargetPixel>(
    commands: &[Command],
    path_geometry: &PhysicalRect,
    clip_rects: &[PhysicalRect],
    color: PremultipliedRgbaColor,
    style: zeno::Style,
    buffers: &mut PathBuffers,
    buffer: &mut impl crate::target_pixel_buffer::TargetPixelBuffer<TargetPixel = T>,
) {
    let path_x = path_geometry.origin.x as i32;
    let path_y = path_geometry.origin.y as i32;
    let path_width = path_geometry.size.width as i32;
    let path_height = path_geometry.size.height as i32;
    if path_width <= 0 || path_height <= 0 || clip_rects.is_empty() {
        return;
    }

    // bounding box of the clip rectangles, inside the path and the buffer (screen coordinates)
    let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for clip in clip_rects {
        x0 = x0.min(clip.origin.x as i32);
        y0 = y0.min(clip.origin.y as i32);
        x1 = x1.max(clip.max_x() as i32);
        y1 = y1.max(clip.max_y() as i32);
    }
    let x0 = x0.max(path_x).max(0);
    let y0 = y0.max(path_y).max(0);
    let x1 = x1.min(path_x + path_width).min(buffer.line_slice(0).len() as i32);
    let y1 = y1.min(path_y + path_height).min(buffer.num_lines() as i32);
    if x0 >= x1 || y0 >= y1 {
        return;
    }

    // patched: a stroke is converted to its outline once, so the bands only rasterize the
    // filled outline instead of stroking the path again for every band
    let PathBuffers { mask: mask_buffer, scratch, outline, subpaths } = buffers;
    let (commands, style) = match style {
        Style::Stroke(_) => {
            outline.clear();
            let fill = zeno::apply(commands, style, None, outline);
            (outline.as_slice(), Style::Fill(fill))
        }
        _ => (commands, style),
    };

    find_subpaths(commands, subpaths);
    // patched: an opaque color is written instead of blended where it covers a pixel completely,
    // like TargetPixel::blend_slice does (the ground of the attitude indicator fills half of it)
    let opaque = (color.alpha == u8::MAX).then(|| T::from_rgb(color.red, color.green, color.blue));

    let width = (x1 - x0) as usize;
    let band_height = (MAX_MASK_BYTES / width).clamp(1, (y1 - y0) as usize);

    let mut band_y = y0;
    while band_y < y1 {
        let height = band_height.min((y1 - band_y) as usize);
        // patched: only the subpaths that reach into the rows of the band, in the coordinates
        // of the path, with a margin of one pixel for anti-aliased edges
        let band_commands = BandCommands {
            commands,
            subpaths,
            top: (band_y - path_y) as f32 - 1.0,
            bottom: (band_y - path_y) as f32 + height as f32 + 1.0,
        };
        if !subpaths.iter().any(|s| s.reaches(band_commands.top, band_commands.bottom)) {
            band_y += height as i32;
            continue;
        }
        let band = &mut mask_buffer[..width * height];
        band.fill(0);

        // Render the band of the path into the mask, the offset moves it to the band origin
        Mask::with_scratch(band_commands, scratch)
            .size(width as u32, height as u32)
            .offset(zeno::Vector::new((path_x - x0) as f32, (path_y - band_y) as f32))
            .style(style)
            .render_into(band, None);

        let band_end = band_y + height as i32;
        for clip in clip_rects {
            let cx0 = (clip.origin.x as i32).max(x0);
            let cx1 = (clip.max_x() as i32).min(x1);
            let cy0 = (clip.origin.y as i32).max(band_y);
            let cy1 = (clip.max_y() as i32).min(band_end);
            if cx0 >= cx1 || cy0 >= cy1 {
                continue;
            }
            let row_width = (cx1 - cx0) as usize;
            for y in cy0..cy1 {
                let mask_row =
                    &band[(y - band_y) as usize * width + (cx0 - x0) as usize..][..row_width];
                let line = &mut buffer.line_slice(y as usize)[cx0 as usize..cx1 as usize];

                // patched: most of a mask is empty, the box of a path is mostly outside its shape.
                // The coverage is read an aligned word (4 pixels) at a time, and only words with
                // coverage are blended pixel by pixel: about a quarter of the loads and branches
                // for empty pixels (the Cortex-M0+ can't load unaligned words).
                let (head, words, tail) = bytemuck::pod_align_to::<u8, u32>(mask_row);
                let (line_head, line) = line.split_at_mut(head.len());
                let (line_words, line_tail) = line.split_at_mut(words.len() * 4);
                for (pixel, &coverage) in line_head.iter_mut().zip(head) {
                    blend_coverage(pixel, coverage, color, opaque);
                }
                for (pixels, &word) in line_words.as_chunks_mut::<4>().0.iter_mut().zip(words) {
                    match (word, opaque) {
                        (0, _) => {}
                        (u32::MAX, Some(opaque)) => pixels.fill(opaque),
                        _ => {
                            for (pixel, coverage) in pixels.iter_mut().zip(word.to_ne_bytes()) {
                                blend_coverage(pixel, coverage, color, opaque);
                            }
                        }
                    }
                }
                for (pixel, &coverage) in line_tail.iter_mut().zip(tail) {
                    blend_coverage(pixel, coverage, color, opaque);
                }
            }
        }

        band_y = band_end;
    }
}

/// Render a filled path
///
/// * `commands` - The path commands to render
/// * `path_geometry` - The full bounding box of the path in screen coordinates
/// * `clip_geometry` - The clipped region where the path should be rendered (intersection of path and clip)
/// * `color` - The color to render the path
/// * `buffer` - The target pixel buffer
pub fn render_filled_path<T: TargetPixel>(
    commands: &[Command],
    path_geometry: &PhysicalRect,
    clip_rects: &[PhysicalRect],
    color: PremultipliedRgbaColor,
    buffer: &mut impl crate::target_pixel_buffer::TargetPixelBuffer<TargetPixel = T>,
) {
    PATH_BUFFERS.with(|buffers| {
        render_path_with_style(
            commands,
            path_geometry,
            clip_rects,
            color,
            zeno::Style::Fill(Fill::NonZero),
            &mut buffers.borrow_mut(),
            buffer,
        )
    });
}

pub struct StrokeStyle<'a> {
    pub width: f32,
    pub line_cap: i_slint_core::items::LineCap,
    pub line_join: i_slint_core::items::LineJoin,
    pub miter_limit: f32,
    pub dash_array: &'a [f32],
    pub dash_offset: f32,
}

/// Render a stroked path
///
/// * `commands` - The path commands to render
/// * `path_geometry` - The full bounding box of the path in screen coordinates
/// * `clip_geometry` - The clipped region where the path should be rendered (intersection of path and clip)
/// * `color` - The color to render the path
/// * `stroke_width` - The width of the stroke
/// * `buffer` - The target pixel buffer
pub fn render_stroked_path<T: TargetPixel>(
    commands: &[Command],
    path_geometry: &PhysicalRect,
    clip_rects: &[PhysicalRect],
    color: PremultipliedRgbaColor,
    stroke_style: StrokeStyle<'_>,
    buffer: &mut impl crate::target_pixel_buffer::TargetPixelBuffer<TargetPixel = T>,
) {
    let mut stroke = Stroke::new(stroke_style.width);
    stroke
        .cap(match stroke_style.line_cap {
            i_slint_core::items::LineCap::Round => Cap::Round,
            i_slint_core::items::LineCap::Square => Cap::Square,
            i_slint_core::items::LineCap::Butt | _ => Cap::Butt,
        })
        .join(match stroke_style.line_join {
            i_slint_core::items::LineJoin::Round => Join::Round,
            i_slint_core::items::LineJoin::Bevel => Join::Bevel,
            i_slint_core::items::LineJoin::Miter | _ => Join::Miter,
        })
        .miter_limit(stroke_style.miter_limit)
        .dash(stroke_style.dash_array, stroke_style.dash_offset);

    let style = Style::Stroke(stroke);
    PATH_BUFFERS.with(|buffers| {
        render_path_with_style(
            commands,
            path_geometry,
            clip_rects,
            color,
            style,
            &mut buffers.borrow_mut(),
            buffer,
        )
    });
}
