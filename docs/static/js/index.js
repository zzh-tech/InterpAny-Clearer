window.HELP_IMPROVE_VIDEOJS = false;


var id_table_0 = ['ct2-CGSVRN', 'np10-CXQFYN', 'x3-VHQTSI', 'x26-FVPOQG', 'x28-LBFFPD', 'np1-JYEFHB', 'x25-YFOSCZ', 'np2-ZBDKCJ', 'np7-LBIDIJ', 'tp1-PAFAEH', 'tp0-RWHPIX', 'x17-JVSNDO', 'x36-RIVJAK', 'x20-VNOEMX', 'ct3-GLGJJW', 'np6-ZRSUNQ', 'x39-HLEYJC', 'x19-PFHBBG', 'x16-CWOEGT'];
var id_table_zhao = ['x14-FJHAPL', 'x5-XDTZRN', 'x9-GHBTYP', 'x1-WDVURU', 'x7-OXFJXM', 'x8-NYGSLL', 'x11-SNSEER', 'x30-OZZPJT', 'x10-UFQKEA', 'x13-ABTDLR', 'x2-JNIGNT', 'x32-PKEGLK'];
var id_table_cmdp = ['8S7R-060329_2', 'VND0-060329_2', 'AMOH-060329_2', 'XWQM-060208_2', 'P1EV-060208_2', 'BHSN-050615_2', '9QAD-050615_2', 'KKMI-060329_2', 'X5FH-050622_2', 'J6XX-060208_2', 'QZGT-060329_2', '5ZAZ-060208_2', 'FHNJ-060329_2', 'RXZN-060208_2', 'WHO1-060329_2', '92RT-050419_2', 'QQXY-060329_2', '3XKB-060208_2', '535X-060329_2', 'JFZC-050419_2', '9UL2-050611_2', 'R77S-060329_2', 'K36K-060208_2', 'ALMI-060208_2', '60R0-060329_2', 'K5AT-050611_2', 'S36F-050611_2', '62RT-060329_2', 'B4IY-050419_2', 'SDCQ-060329_2', '73MC-050615_2', 'KN46-060208_2', 'Z7QW-060208_2', 'T6D7-060329_2', '7FNY-060329_2', 'MSX9-060329_2', 'ZBZP-050611_2', 'BUP6-060329_2', 'UYGI-060329_2', '7IVG-060208_2', 'NRRD-060208_2', 'ENGX-060208_2', 'VIMC-050615_2'];
// '23UG-060329_2',
// '1FPP-060329_2', 'WWGO-060329_2', '7WUF-060329_2', 'QHTL-060329_2', 'FPOG-060329_2', 'XOPC-060329_2',  'YWXB-060329_2',

$(document).ready(function () {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function () {
        // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
        $(".navbar-burger").toggleClass("is-active");
        $(".navbar-menu").toggleClass("is-active");

    });


    /*
    var options = {
      slidesToScroll: 1,
      slidesToShow: 3,
      loop: true,
      infinite: true,
      autoplay: false,
      autoplaySpeed: 3000,
    }

    // Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
      // Add listener to  event
      carousels[i].on('before:show', state => {
        console.log(state);
      });
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
      // bulmaCarousel instance is available as element.bulmaCarousel
      element.bulmaCarousel.on('before-show', function(state) {
        console.log(state);
      });
    }
    */


    $('.results-carousel').slick({
        dots: true,
        infinite: true,
        speed: 300,
        slidesToShow: 1,
        autoplay: true,
        centerMode: true,
        autoplaySpeed: 3000,
        variableWidth: true
    });

    $('.cmdp-carousel').slick({
        // dots: true,
        infinite: true,
        speed: 500,
        slidesToShow: 1,
        autoplay: false,
        // autoplaySpeed: 3000,
        centerMode: true,
        variableWidth: true
    });


    for (let i = 1; i < 4; i++) {
        $('#face-slider').on('input', function (event) {
            updateScene(this.value, '#focal_grid' + String(i), i, $('#focal-slider-wild').val());
        });
    }
    $('#face-slider').on('input', function (event) {
        updateScene(this.value, '#focal_grid0', 0, $('#focal-slider-wild').val());
    });
    $('#face-slider-zhao').on('input', function (event) {
        updateSceneZhao(this.value, '#zhao_grid0', 0, $('#focal-slider-zhao').val());
    });
    for (let i = 1; i < 4; i++) {
        $('#face-slider-zhao').on('input', function (event) {
            updateSceneZhao(this.value, '#zhao_grid' + String(i), i, $('#focal-slider-zhao').val());
        });
    }
    for (let i = 0; i < 4; i++) {
        $('#face-slider-cmdp').on('input', function (event) {
            updateSceneCMDP(this.value, '#cmdp_grid' + String(i), i);
        });
    }


    $('#focal-slider-wild').on('input', function (event) {
        updateImageWhole($('#focal-slider-wild').val(), '#focal_grid1');
    });
    $('#focal-slider-wild').on('input', function (event) {
        updateImageWhole($('#focal-slider-wild').val(), '#focal_grid2');
    });
    $('#focal-slider-wild').on('input', function (event) {
        updateImageWhole($('#focal-slider-wild').val(), '#focal_grid3');
    });
    $('#focal-slider-zhao').on('input', function (event) {
        updateImageWhole($('#focal-slider-zhao').val(), '#zhao_grid1');
    });

    bulmaSlider.attach();


    for (let i = 1; i < 4; i++) {
        updateScene($('#face-slider').val(), '#focal_grid' + String(i), i, $('#focal-slider-wild').val());
    }
    updateScene($('#face-slider').val(), '#focal_grid0', 0, $('#focal-slider-wild').val());

    for (let i = 0; i < 4; i++) {
        updateSceneCMDP($('#face-slider').val(), '#cmdp_grid' + String(i), i);
    }
    updateImageWhole($('#focal-slider-wild').val(), '#focal_grid1');
    updateImageWhole($('#focal-slider-wild').val(), '#focal_grid2');
    updateImageWhole($('#focal-slider-zhao').val(), '#zhao_grid1');
})

window.onresize = function () {
    for (let i = 1; i < 4; i++) {
        updateScene($('#face-slider').val(), '#focal_grid' + String(i), i, $('#focal-slider-wild').val());
    }
    updateScene($('#face-slider').val(), '#focal_grid0', 0, $('#focal-slider-wild').val());

    for (let i = 0; i < 4; i++) {
        updateSceneCMDP($('#face-slider').val(), '#cmdp_grid' + String(i), i);
    }
    updateImageWhole($('#focal-slider-wild').val(), '#focal_grid1');
    updateImageWhole($('#focal-slider-wild').val(), '#focal_grid2');
    updateImageWhole($('#focal-slider-zhao').val(), '#zhao_grid1');
};


$(window).on("load", function () {
    for (let i = 1; i < 4; i++) {
        updateScene($('#face-slider').val(), '#focal_grid' + String(i), i, $('#focal-slider-wild').val());
    }
    updateScene($('#face-slider').val(), '#focal_grid0', 0, $('#focal-slider-wild').val());

    for (let i = 0; i < 4; i++) {
        updateSceneCMDP($('#face-slider').val(), '#cmdp_grid' + String(i), i);
    }
    updateImageWhole($('#focal-slider-wild').val(), '#focal_grid1');
    updateImageWhole($('#focal-slider-wild').val(), '#focal_grid2');
    updateImageWhole($('#focal-slider-wild').val(), '#focal_grid3');

    updateImageWhole($('#focal-slider-zhao').val(), '#zhao_grid1');

    // Reset gifs once everything is loaded to synchronize playback.
    $('.preload').attr('src', function (i, a) {
        $(this).attr('src', '').removeClass('preload').attr('src', a);
    });


});

Number.prototype.clamp = function (min, max) {
    return Math.min(Math.max(this, min), max);
};


function updateScene(value, tag, k, focal) {
    // 'tp3-CTTSKA', 
    // 'tp4-DOEGEL', 
    // 'x35-JFUJTJ'
    // 'tp0-MSBXWE', 'np9-EYBFBI', 'np16-SPKCNT', 'np9-EYBFBI',
    // var id_table_0 = ['ct2-CGSVRN', 'np10-CXQFYN', 'x3-VHQTSI', 'ct1-DIVUYD', 'x44-BOBYEY',  'x26-FVPOQG', 'x28-LBFFPD', 'np1-JYEFHB', 'x25-YFOSCZ', 'np2-ZBDKCJ', 'np7-LBIDIJ', 'tp1-PAFAEH', 'tp0-RWHPIX', 'x17-JVSNDO', 'x36-RIVJAK', 'x20-VNOEMX', 'ct3-GLGJJW', 'np6-ZRSUNQ', 'x39-HLEYJC', 'x19-PFHBBG', 'x16-CWOEGT'];
    var id_table = id_table_0;
    // var methods_list = ['','_ours_dolly','_fried_dolly','_ref']
    var methods_list = ['', '_ours', '_fried', '_wacv_dolly'];
    //$(tag).height(Math.round($(tag).height()))
    // width = $(tag)[0].getBoundingClientRect().width
    // naturalwidth= $(tag)[0].naturalWidth;
    // num_images = naturalwidth/512.0;
    // console.log(width);
    // console.log(naturalwidth);
    console.log(value);
    // let left = value * width / num_images;
    // console.log(left);
    $(tag).attr('src', "./static/images/Wild/" + id_table[value] + methods_list[k] + ".jpg");
    // $('#sequence_name').attr('innerHTML', ": "+id_table[value]+"&nbsp;");
    // document.getElementById("sequence_name").innerHTML = id_table[value].slice(0, -2);+"&nbsp;";
    // document.getElementById("frame-idx").innerHTML =  ("00"+String(Number(value)+1)).slice(-2);
}

function updateSceneCMDP(value, tag, k) {
    // 'tp3-CTTSKA', 
    // 'tp4-DOEGEL', 
    // 'x35-JFUJTJ'
    // 'tp0-MSBXWE', 'np9-EYBFBI', 'np16-SPKCNT', 'np9-EYBFBI',
    // var id_table_0 = ['ct2-CGSVRN', 'np10-CXQFYN', 'x3-VHQTSI', 'ct1-DIVUYD', 'x44-BOBYEY',  'x26-FVPOQG', 'x28-LBFFPD', 'np1-JYEFHB', 'x25-YFOSCZ', 'np2-ZBDKCJ', 'np7-LBIDIJ', 'tp1-PAFAEH', 'tp0-RWHPIX', 'x17-JVSNDO', 'x36-RIVJAK', 'x20-VNOEMX', 'ct3-GLGJJW', 'np6-ZRSUNQ', 'x39-HLEYJC', 'x19-PFHBBG', 'x16-CWOEGT'];
    var id_table = id_table_cmdp;
    // var methods_list = ['','_ours_dolly','_fried_dolly','_ref']
    var methods_list = ['', '_fried', '_ours', '_ref16'];
    //$(tag).height(Math.round($(tag).height()))
    // width = $(tag)[0].getBoundingClientRect().width
    // naturalwidth= $(tag)[0].naturalWidth;
    // num_images = naturalwidth/512.0;
    // console.log(width);
    // console.log(naturalwidth);
    console.log(value);
    // let left = value * width / num_images;
    // console.log(left);
    $(tag).attr('src', "./static/images/CMDP/");
    $(tag).attr('src', "./static/images/CMDP/" + id_table[value] + methods_list[k] + ".jpg");
    // $('#sequence_name').attr('innerHTML', ": "+id_table[value]+"&nbsp;");
    // document.getElementById("sequence_name").innerHTML = id_table[value].slice(0, -2);+"&nbsp;";
    // document.getElementById("frame-idx").innerHTML =  ("00"+String(Number(value)+1)).slice(-2);
}


function updateImageWhole(value, tag) {
    //$(tag).height(Math.round($(tag).height()))
    width = $(tag)[0].getBoundingClientRect().width
    naturalwidth = $(tag)[0].naturalWidth;
    num_images = naturalwidth / 512.0;
    // console.log(width);
    // console.log(naturalwidth);
    console.log(value);
    let left = value * width / num_images;
    console.log(left);
    $(tag).css('left', -left + 'px');
}

function updateSceneZhao(value, tag, k) {
    var id_table = id_table_zhao;
    // var methods_list = ['','_ours_dolly','_fried_dolly','_ref']
    var methods_list = ['', '_ours', '_Fried', '_Zhao']
    //$(tag).height(Math.round($(tag).height()))
    // width = $(tag)[0].getBoundingClientRect().width
    // naturalwidth= $(tag)[0].naturalWidth;
    // num_images = naturalwidth/512.0;
    // console.log(width);
    // console.log(naturalwidth);
    // console.log(value);
    // let left = value * width / num_images;
    // console.log(left);
    $(tag).attr('src', "./static/images/Zhao/" + id_table[value] + methods_list[k] + ".jpg");
    // $('#sequence_name').attr('innerHTML', ": "+id_table[value]+"&nbsp;");
    // document.getElementById("sequence_name").innerHTML = id_table[value].slice(0, -2);+"&nbsp;";
    // document.getElementById("frame-idx").innerHTML =  ("00"+String(Number(value)+1)).slice(-2);
}