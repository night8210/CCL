<?php
include "./layout/header.php";
include "./layout/navbar.php";
include "./class/database.php";
include "./class/city.php";
$city=new city();
$city->id=$_GET["id"];
$city->get_one_city();
if(isset($_POST["submit"]))
{
    $city->city_name=$_POST["city_name"];
    $city->population=$_POST["population"];
    $city->country_code=$_POST["country_code"];
}
?>
<div class="container my-5">
    <div class="row justify-content-center">
        <div class="col-md-6 col-xs-12">
            <form action='./edit.php?id=<?=$_GET["id"]?>' method="post">
                <div class="form-group">
                    <label for="city_name">城市名字</label>
                    <input type="text" class="form-control" name="city_name" id="city_name" required value="<?=$city->city_name?>">
                </div>
                <div class="form-group">
                    <label for="population">城市人口</label>
                    <input type="text" class="form-control" name="population" id="population" required value="<?=$city->population?>">
                </div>
                <div class="form-group">
                    <label for="country_code">國家代碼</label>
                    <select class="form-control" name="country_code" id="country_code">
                    <?php
                    include "./class/country.php";
                    $country=new country();
                    $data=$country->get_all_country_code();
                    ?>
                    <?php foreach($data as $row):?>
                    <?php if($row["country_code"]==$city->country_code):?>
                        <option value="<?= $row["country_code"]?>" selected>
                            <?= $row["country_code"]?>
                        </option>
                    <?php else:?>
                        <option value="<?= $row["country_code"]?>">
                            <?= $row["country_code"]?>
                        </option>
                    <?php endif;?>
                    <?php endforeach;?>
                    </select>
                </div>
                <button type="submit" name="submit" class="btn btn-info">提交資料</button>
                <a href="./index.php" class="btn btn-outline-secondary">返回首頁</a>
            </form>
            <?php
            if(isset($_POST["submit"]))
            {
                if($city->update())
                {
                    echo '<div class="alert alert-warning my-3" role="alert">編輯成功</div>';
                }
                else
                {
                    echo '<div class="alert alert-warning my-3" role="alert">編輯失敗，稍後再試</div>';
                }
            }
            ?>
        </div>
    </div>
</div>