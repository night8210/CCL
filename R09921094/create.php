<?php
include "./layout/header.php";
include "./layout/navbar.php";
include "./class/database.php";
?>
<div class="container my-5">
    <div class="row justify-content-center">
        <div class="col-md-6 col-xs-12">
            <form action="./create.php" method="post">
                <div class="form-group">
                    <label for="city_name">城市名字</label>
                    <input type="text" class="form-control" name="city_name" id="city_name" required>
                </div>
                <div class="form-group">
                    <label for="population">城市人口</label>
                    <input type="text" class="form-control" name="population" id="population" required>
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
                        <option value="<?= $row["country_code"]?>">
                            <?= $row["country_code"]?>
                        </option>
                    <?php endforeach;?>
                    </select>
                </div>
                <button type="submit" name="submit" class="btn btn-info">提交資料</button>
                <a href="./index.php" class="btn btn-outline-secondary">返回首頁</a>
            </form>
            <?php
            if(isset($_POST["submit"]))
            {
                include "./class/city.php";
                $city=new city();
                $city->city_name=$_POST["city_name"];
                $city->population=$_POST["population"];
                $city->country_code=$_POST["country_code"];
                if($city->create())
                {
                    echo '<div class="alert alert-warning my-3" role="alert">新增成功</div>';
                }
                else
                {
                    echo '<div class="alert alert-warning my-3" role="alert">新增失敗，稍後再試</div>';
                }
            }
            ?>
        </div>
    </div>
</div>