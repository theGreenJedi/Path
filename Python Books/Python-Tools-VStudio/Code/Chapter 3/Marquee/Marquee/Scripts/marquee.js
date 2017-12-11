$.fn.textwidth = function () {
    var calc = '<span style="display:none">' + $(this).text() + '</span>';
    $('body').append(calc);
    var width = $('body').find('span:last').width();
    $('body').find('span:last').remove();
    return width;

};

$.fn.marquee = function () {
    var that = $(this),
    calculatedWidth = that.textwidth(),
    offset = calculatedWidth,
    width = offset,
    css = {
        'text-indent': that.css('text-indent'),
        'overflow': that.css('overflow'),
        'white-space': that.css('white-space')
    },
    marqueeCss = {
        'text-ident': width,
        'overflow': 'hidden',
        'white-space': 'nowrap'
    };

    function go() {
        if (width == (calculatedWidth * -1)) {
            width = offset;
        }
        that.css('text-indent', width + 'px');
        width--;
        setTimeout(go, 1e1);
    };
    that.css(marqueeCss);
    width--;
    go();
};
