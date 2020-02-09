function showdets(im, boxes, names, ids, out)
% Draw bounding boxes on top of an image.
%   showboxes(im, boxes, out)
%
%   If out is given, a pdf of the image is generated (requires export_fig).

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2011-2012 Ross Girshick
% Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
% Copyright (C) 2007 Pedro Felzenszwalb, Deva Ramanan
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

if nargin > 4
  % different settings for producing pdfs
  toprint = true;
  cwidth = 2;
  wwidth = cwidth + 1.1;
else
  toprint = false;
  cwidth = 2;
end

if toprint
    h = figure('visible', 'off');
    image(im);
    truesize(h);
else
    h = figure;
    image(im);
end

axis image;
axis off;
set(h, 'Color', 'white');

if ~isempty(boxes)
  numfilters = size(boxes,1);
  
  % draw the boxes with the detection window on top (reverse order)
  for i = numfilters:-1:1
    x1 = boxes(i,1);
    y1 = boxes(i,2);
    x2 = boxes(i,3);
    y2 = boxes(i,4);
    % remove unused filters
    del = find(((x1 == 0) .* (x2 == 0) .* (y1 == 0) .* (y2 == 0)) == 1);
    x1(del) = [];
    x2(del) = [];
    y1(del) = [];
    y2(del) = [];
    if ids(i) > 200 
      c = 'r';
      s = '-';
    else
      c = 'b';
      s = '-';
    end
    line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', c, 'linewidth', cwidth, 'linestyle', s);
    ss = regexp(names{i}, ',', 'split');
    x_t = double(max(5, x1-5)); y_t = double(min(y2+5, size(im, 1)-15));
    text(x_t,y_t,sprintf('%s: %2.1f', ss{1}, boxes(i,5)),...
        'BackgroundColor', [0.7 0.9 0.7], 'FontSize', 20, 'Color', c);
  end
end

if toprint
  set(gca, 'LooseInset', get(gca, 'TightInset'));
  saveas(h, out);
end
