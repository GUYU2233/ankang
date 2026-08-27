<template>
  <el-container class="app-root">
    <el-aside width="220px" class="sidebar">
      <div class="logo">
        <span class="logo-main">智护安康</span>
        <span class="logo-sub">跌倒风险智能预警</span>
      </div>
      <el-menu :default-active="$route.path" router background-color="#001529" text-color="#d0d8e8" active-text-color="#67c23a">
        <el-menu-item index="/dashboard">数据看板</el-menu-item>
        <el-menu-item index="/devices">设备接入</el-menu-item>
        <el-menu-item index="/residents">老人档案</el-menu-item>
        <el-menu-item index="/live">实时画面</el-menu-item>
        <el-menu-item index="/alerts">告警中心</el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header class="topbar">
        <div class="page-title">{{ $route.meta.title || '' }}</div>
        <el-tag type="success" effect="dark" size="small">萤石开放平台 + 海康兼容</el-tag>
      </el-header>
      <el-main class="main-body">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
</script>

<style>
html, body, #app { height: 100%; margin: 0; padding: 0; background: #f3f5f9; }
.app-root { height: 100%; }
.sidebar { background: #001529; }
.logo { padding: 18px 14px; color: #fff; }
.logo-main { display: block; font-size: 19px; font-weight: 700; }
.logo-sub { font-size: 12px; color: #9fb0c8; }
.topbar { display: flex; align-items: center; justify-content: space-between; background: #fff; border-bottom: 1px solid #e4e7ed; }
.page-title { font-size: 18px; font-weight: 600; }
.main-body { padding: 16px; }
</style>